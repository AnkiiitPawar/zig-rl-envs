//! CartPole-v1 (Barto, Sutton and Anderson, 1983), as Gymnasium 1.4.0
//! defines it, for many environments at once.
//!
//! The state is kept in float64, one array per variable, and every
//! operation is done in the order Gymnasium does it, so that given the same
//! state and action the next state, observation, reward and termination are
//! bit-identical. Observations are float32, as Gymnasium returns them.

const std = @import("std");
const libm = @import("libm.zig");
const batch = @import("batch.zig");

pub const obs_dim = 4;
pub const state_dim = 4;
pub const n_actions = 2;
pub const max_steps = 500;

const gravity: f64 = 9.8;
const masscart: f64 = 1.0;
const masspole: f64 = 0.1;
const total_mass: f64 = masspole + masscart;
const length: f64 = 0.5;
const polemass_length: f64 = masspole * length;
const force_mag: f64 = 10.0;
const tau: f64 = 0.02;
const pi: f64 = std.math.pi;
// Each operation rounded to float64 in Python's order: (12 * 2) * pi / 360.
const theta_threshold: f64 = 24.0 * pi / 360.0;
const x_threshold: f64 = 2.4;
const four_thirds: f64 = 4.0 / 3.0;

pub const Action = u8;

/// One step of the dynamics for a single environment: the new state and
/// whether it is terminal. `s` is x, x_dot, theta, theta_dot.
pub fn dynamics(s: [4]f64, action: Action, cos_t: f64, sin_t: f64) struct { s: [4]f64, terminated: bool } {
    const x = s[0];
    const x_dot = s[1];
    const theta = s[2];
    const theta_dot = s[3];
    const force: f64 = if (action == 1) force_mag else -force_mag;
    const temp = (force + polemass_length * (theta_dot * theta_dot) * sin_t) / total_mass;
    const thetaacc = (gravity * sin_t - cos_t * temp) / (length * (four_thirds - masspole * (cos_t * cos_t) / total_mass));
    const xacc = temp - polemass_length * thetaacc * cos_t / total_mass;
    const n: [4]f64 = .{ x + tau * x_dot, x_dot + tau * xacc, theta + tau * theta_dot, theta_dot + tau * thetaacc };
    const terminated = n[0] < -x_threshold or n[0] > x_threshold or n[2] < -theta_threshold or n[2] > theta_threshold;
    return .{ .s = n, .terminated = terminated };
}

pub fn observe(s: [4]f64) [obs_dim]f32 {
    return .{ @floatCast(s[0]), @floatCast(s[1]), @floatCast(s[2]), @floatCast(s[3]) };
}

/// Gymnasium's reset distribution: each variable uniform on [-0.05, 0.05).
pub fn initial(rand: std.Random) [4]f64 {
    var s: [4]f64 = undefined;
    for (&s) |*v| v.* = -0.05 + 0.1 * rand.float(f64);
    return s;
}

/// Many environments, stepped together. The arithmetic runs on 8 lanes at a
/// time; sin and cos come from the C library, one value at a time.
pub const Batch = struct {
    n: usize,
    s: [4][]f64, // x, x_dot, theta, theta_dot
    trig: [2][]f64, // cos theta, sin theta
    act: []f64, // +1 or -1
    steps: []u32,

    pub fn init(gpa: std.mem.Allocator, n: usize) !Batch {
        const padded = batch.pad(n);
        var b: Batch = .{ .n = n, .s = undefined, .trig = undefined, .act = try gpa.alloc(f64, padded), .steps = try gpa.alloc(u32, n) };
        for (&b.s) |*a| a.* = try gpa.alloc(f64, padded);
        for (&b.trig) |*a| a.* = try gpa.alloc(f64, padded);
        for (b.s) |a| @memset(a, 0);
        @memset(b.act, 1);
        @memset(b.steps, 0);
        return b;
    }

    pub fn deinit(b: *Batch, gpa: std.mem.Allocator) void {
        for (b.s) |a| gpa.free(a);
        for (b.trig) |a| gpa.free(a);
        gpa.free(b.act);
        gpa.free(b.steps);
    }

    pub fn get(b: *const Batch, i: usize) [4]f64 {
        return .{ b.s[0][i], b.s[1][i], b.s[2][i], b.s[3][i] };
    }

    pub fn set(b: *Batch, i: usize, s: [4]f64) void {
        for (0..4) |k| b.s[k][i] = s[k];
        b.steps[i] = 0;
    }

    /// Advances every environment by one step. Writes rewards and
    /// termination flags; the caller handles resets (see batch.zig).
    pub fn advance(b: *Batch, actions: []const Action, reward: []f64, terminated: []bool) void {
        const n = b.n;
        for (0..n) |i| {
            b.trig[0][i] = libm.cos(b.s[2][i]);
            b.trig[1][i] = libm.sin(b.s[2][i]);
            b.act[i] = if (actions[i] == 1) 1 else -1;
        }
        const L = batch.lanes;
        const V = @Vector(L, f64);
        var i: usize = 0;
        while (i < n) : (i += L) {
            const x = batch.load(b.s[0], i);
            const x_dot = batch.load(b.s[1], i);
            const theta = batch.load(b.s[2], i);
            const theta_dot = batch.load(b.s[3], i);
            const c = batch.load(b.trig[0], i);
            const s = batch.load(b.trig[1], i);
            const force = batch.load(b.act, i) * @as(V, @splat(force_mag));
            const temp = (force + @as(V, @splat(polemass_length)) * (theta_dot * theta_dot) * s) / @as(V, @splat(total_mass));
            const thetaacc = (@as(V, @splat(gravity)) * s - c * temp) /
                (@as(V, @splat(length)) * (@as(V, @splat(four_thirds)) - @as(V, @splat(masspole)) * (c * c) / @as(V, @splat(total_mass))));
            const xacc = temp - @as(V, @splat(polemass_length)) * thetaacc * c / @as(V, @splat(total_mass));
            const t: V = @splat(tau);
            batch.store(b.s[0], i, x + t * x_dot);
            batch.store(b.s[1], i, x_dot + t * xacc);
            batch.store(b.s[2], i, theta + t * theta_dot);
            batch.store(b.s[3], i, theta_dot + t * thetaacc);
        }
        for (0..n) |k| {
            const x = b.s[0][k];
            const th = b.s[2][k];
            terminated[k] = x < -x_threshold or x > x_threshold or th < -theta_threshold or th > theta_threshold;
            reward[k] = 1.0;
            b.steps[k] += 1;
        }
    }

    pub fn observeInto(b: *const Batch, i: usize, out: []f32) void {
        for (0..4) |k| out[k] = @floatCast(b.s[k][i]);
    }
};

const testing = std.testing;

test "batch and single-environment dynamics agree bit for bit" {
    const gpa = testing.allocator;
    const n = 37;
    var b = try Batch.init(gpa, n);
    defer b.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(3);
    const r = prng.random();
    var acts: [n]u8 = undefined;
    var reward: [n]f64 = undefined;
    var term: [n]bool = undefined;
    for (0..n) |i| {
        b.set(i, .{ r.float(f64) * 4 - 2, r.float(f64) * 4 - 2, r.float(f64) * 0.4 - 0.2, r.float(f64) * 4 - 2 });
        acts[i] = r.uintLessThan(u8, 2);
    }
    var before: [n][4]f64 = undefined;
    for (0..n) |i| before[i] = b.get(i);
    b.advance(&acts, &reward, &term);
    for (0..n) |i| {
        const s = before[i];
        const one = dynamics(s, acts[i], libm.cos(s[2]), libm.sin(s[2]));
        const got = b.get(i);
        for (0..4) |k| try testing.expectEqual(one.s[k], got[k]);
        try testing.expectEqual(one.terminated, term[i]);
    }
}

test "the pole falls without control" {
    var s: [4]f64 = .{ 0, 0, 0.01, 0 };
    var steps: usize = 0;
    while (steps < 1000) : (steps += 1) {
        const r = dynamics(s, @intCast(steps % 2), libm.cos(s[2]), libm.sin(s[2]));
        s = r.s;
        if (r.terminated) break;
    }
    try testing.expect(steps < 200);
}
