//! Pendulum-v1, as Gymnasium 1.4.0 defines it, for many environments at once.
//!
//! The torque arrives as float32, as Gymnasium's action space gives it, and
//! the arithmetic follows NumPy's rules for mixing float32 with float64: the
//! torque's own products stay in float32, everything else is float64. Squares
//! in the cost are NumPy's `x**2`, that is pow. That is what makes the results
//! bit-identical to Gymnasium's.

const std = @import("std");
const libm = @import("libm.zig");
const batch = @import("batch.zig");

pub const obs_dim = 3;
pub const state_dim = 2;
pub const act_dim = 1;
pub const max_steps = 200;
pub const max_torque: f32 = 2.0;

const max_speed: f64 = 8.0;
const dt: f64 = 0.05;
const g: f64 = 10.0;
const m: f64 = 1.0;
const l: f64 = 1.0;
const pi: f64 = std.math.pi;
// 3 * g / (2 * l), and 3.0 / (m * l**2), in Python's order.
const gravity_coef: f64 = 3.0 * g / (2.0 * l);
const torque_coef: f32 = @floatCast(3.0 / (m * (l * l)));
const torque_cost: f32 = 0.001;

pub const Action = f32;

fn angleNormalize(x: f64) f64 {
    return libm.pyMod(x + pi, 2.0 * pi) - pi;
}

/// One step: the new state (theta, theta_dot) and the reward.
pub fn dynamics(s: [2]f64, action: Action) struct { s: [2]f64, reward: f64 } {
    const th = s[0];
    const thdot = s[1];
    const u = std.math.clamp(action, -max_torque, max_torque);
    const costs = cost(th, thdot, u);
    var newthdot = thdot + (gravity_coef * libm.sin(th) + @as(f64, torque_coef * u)) * dt;
    newthdot = std.math.clamp(newthdot, -max_speed, max_speed);
    const newth = th + newthdot * dt;
    return .{ .s = .{ newth, newthdot }, .reward = -costs };
}

/// angle_normalize(th)**2 + 0.1 * thdot**2 + 0.001 * (u**2), with u in float32.
fn cost(th: f64, thdot: f64, u: f32) f64 {
    return libm.sq(angleNormalize(th)) + 0.1 * libm.sq(thdot) + @as(f64, torque_cost * libm.sqf(u));
}

pub fn observe(s: [2]f64) [obs_dim]f32 {
    return .{ @floatCast(libm.cos(s[0])), @floatCast(libm.sin(s[0])), @floatCast(s[1]) };
}

/// Gymnasium's reset distribution: theta uniform on [-pi, pi), the angular
/// velocity uniform on [-1, 1).
pub fn initial(rand: std.Random) [2]f64 {
    return .{ -pi + 2 * pi * rand.float(f64), -1.0 + 2.0 * rand.float(f64) };
}

pub const Batch = struct {
    n: usize,
    s: [2][]f64,
    sin_th: []f64,
    tq: []f64, // float32 torque term, widened
    steps: []u32,

    pub fn init(gpa: std.mem.Allocator, n: usize) !Batch {
        const p = batch.pad(n);
        var b: Batch = .{ .n = n, .s = undefined, .sin_th = try gpa.alloc(f64, p), .tq = try gpa.alloc(f64, p), .steps = try gpa.alloc(u32, n) };
        for (&b.s) |*a| a.* = try gpa.alloc(f64, p);
        for (b.s) |a| @memset(a, 0);
        @memset(b.sin_th, 0);
        @memset(b.tq, 0);
        @memset(b.steps, 0);
        return b;
    }

    pub fn deinit(b: *Batch, gpa: std.mem.Allocator) void {
        for (b.s) |a| gpa.free(a);
        gpa.free(b.sin_th);
        gpa.free(b.tq);
        gpa.free(b.steps);
    }

    pub fn get(b: *const Batch, i: usize) [2]f64 {
        return .{ b.s[0][i], b.s[1][i] };
    }

    pub fn set(b: *Batch, i: usize, s: [2]f64) void {
        b.s[0][i] = s[0];
        b.s[1][i] = s[1];
        b.steps[i] = 0;
    }

    /// The library calls (sin, pow) and the cost one value at a time; the
    /// state update on 8 lanes.
    pub fn advance(b: *Batch, actions: []const Action, reward: []f64, terminated: []bool) void {
        const n = b.n;
        for (0..n) |i| {
            const th = b.s[0][i];
            const u = std.math.clamp(actions[i], -max_torque, max_torque);
            b.sin_th[i] = libm.sin(th);
            b.tq[i] = torque_coef * u;
            reward[i] = -cost(th, b.s[1][i], u);
            terminated[i] = false;
            b.steps[i] += 1;
        }
        const L = batch.lanes;
        const sp = batch.splat;
        var i: usize = 0;
        while (i < n) : (i += L) {
            const th = batch.load(b.s[0], i);
            const thdot = batch.load(b.s[1], i);
            var newthdot = thdot + (sp(gravity_coef) * batch.load(b.sin_th, i) + batch.load(b.tq, i)) * sp(dt);
            newthdot = @min(@max(newthdot, sp(-max_speed)), sp(max_speed));
            batch.store(b.s[0], i, th + newthdot * sp(dt));
            batch.store(b.s[1], i, newthdot);
        }
    }

    pub fn observeInto(b: *const Batch, i: usize, out: []f32) void {
        const o = observe(b.get(i));
        @memcpy(out[0..obs_dim], &o);
    }
};

const testing = std.testing;

test "batch and single-environment dynamics agree bit for bit" {
    const gpa = testing.allocator;
    const n = 21;
    var b = try Batch.init(gpa, n);
    defer b.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(4);
    const r = prng.random();
    var acts: [n]f32 = undefined;
    var reward: [n]f64 = undefined;
    var term: [n]bool = undefined;
    for (0..n) |i| {
        b.set(i, .{ r.float(f64) * 20 - 10, r.float(f64) * 16 - 8 });
        acts[i] = r.float(f32) * 6 - 3;
    }
    var before: [n][2]f64 = undefined;
    for (0..n) |i| before[i] = b.get(i);
    b.advance(&acts, &reward, &term);
    for (0..n) |i| {
        const one = dynamics(before[i], acts[i]);
        const got = b.get(i);
        try testing.expectEqual(one.s[0], got[0]);
        try testing.expectEqual(one.s[1], got[1]);
        try testing.expectEqual(one.reward, reward[i]);
    }
}

test "angle normalisation follows Python's remainder" {
    try testing.expectApproxEqAbs(@as(f64, 0.5), angleNormalize(0.5 + 4 * pi), 1e-12);
    try testing.expectApproxEqAbs(@as(f64, -0.5), angleNormalize(-0.5 - 4 * pi), 1e-12);
    try testing.expect(angleNormalize(pi) == -pi);
}
