//! Acrobot-v1 (Sutton, 1996; the "book" dynamics), as Gymnasium 1.4.0
//! defines it, for many environments at once.
//!
//! One step is one fourth-order Runge-Kutta step of 0.2 s. As in Gymnasium,
//! the starting state is drawn in float32, the dynamics run in float64,
//! squares of variables are NumPy's `x**2` (pow), and every operation follows
//! Gymnasium's order, so the trajectories are bit-identical given the same
//! start and actions.

const std = @import("std");
const libm = @import("libm.zig");
const batch = @import("batch.zig");

pub const obs_dim = 6;
pub const state_dim = 4;
pub const n_actions = 3;
pub const max_steps = 500;

const pi: f64 = std.math.pi;
const half_pi: f64 = pi / 2.0;
const dt: f64 = 0.2;
const dt2: f64 = dt / 2.0;
const dt6: f64 = dt / 6.0;
const max_vel_1: f64 = 4.0 * pi;
const max_vel_2: f64 = 9.0 * pi;
const two_pi: f64 = pi - (-pi);

// Gymnasium's link constants, combined in the order its code combines them.
const m1: f64 = 1.0;
const m2: f64 = 1.0;
const l1: f64 = 1.0;
const lc1: f64 = 0.5;
const lc2: f64 = 0.5;
const inertia: f64 = 1.0;
const g: f64 = 9.8;
const d1_a: f64 = m1 * (lc1 * lc1); // m1 * lc1**2
const d1_b: f64 = l1 * l1 + lc2 * lc2; // l1**2 + lc2**2
const d1_c: f64 = 2.0 * l1 * lc2; // 2 * l1 * lc2
const d2_a: f64 = lc2 * lc2; // lc2**2
const d2_b: f64 = l1 * lc2; // l1 * lc2
const phi2_c: f64 = m2 * lc2 * g; // m2 * lc2 * g
const phi1_a: f64 = -m2 * l1 * lc2; // -m2 * l1 * lc2
const phi1_b: f64 = 2.0 * m2 * l1 * lc2; // 2 * m2 * l1 * lc2
const phi1_c: f64 = (m1 * lc1 + m2 * l1) * g; // (m1 * lc1 + m2 * l1) * g
const dd2_a: f64 = m2 * l1 * lc2; // m2 * l1 * lc2
const dd2_den: f64 = m2 * (lc2 * lc2) + inertia; // m2 * lc2**2 + I2

pub const Action = u8;

const torques = [3]f64{ -1.0, 0.0, 1.0 };

/// What Gymnasium's _dsdt needs from the C library at a stage: cos and sin
/// of the angles, and the squares (pow) of the velocities and of d2.
const Lib = struct { c2: f64, s2: f64, c12: f64, c1: f64, dth1_sq: f64, dth2_sq: f64, d2: f64, d2_sq: f64 };

fn d2Of(c2: f64) f64 {
    return m2 * (d2_a + d2_b * c2) + inertia;
}

fn lib(y: [4]f64) Lib {
    const c2 = libm.cos(y[1]);
    const d2 = d2Of(c2);
    return .{ .c2 = c2, .s2 = libm.sin(y[1]), .c12 = libm.cos(y[0] + y[1] - half_pi), .c1 = libm.cos(y[0] - half_pi), .dth1_sq = libm.sq(y[2]), .dth2_sq = libm.sq(y[3]), .d2 = d2, .d2_sq = libm.sq(d2) };
}

/// Gymnasium's _dsdt: the time derivative of (theta1, theta2, dtheta1,
/// dtheta2) under torque `a`.
fn dsdt(y: [4]f64, a: f64, t: Lib) [4]f64 {
    const dth1 = y[2];
    const dth2 = y[3];
    const d1 = d1_a + m2 * (d1_b + d1_c * t.c2) + inertia + inertia;
    const d2 = t.d2;
    const phi2 = phi2_c * t.c12;
    const phi1 = phi1_a * t.dth2_sq * t.s2 - phi1_b * dth2 * dth1 * t.s2 + phi1_c * t.c1 + phi2;
    const ddth2 = (a + d2 / d1 * phi1 - dd2_a * t.dth1_sq * t.s2 - phi2) / (dd2_den - t.d2_sq / d1);
    const ddth1 = -(d2 * ddth2 + phi1) / d1;
    return .{ dth1, dth2, ddth1, ddth2 };
}

fn wrap(x: f64) f64 {
    var v = x;
    while (v > pi) v = v - two_pi;
    while (v < -pi) v = v + two_pi;
    return v;
}

fn bound(x: f64, hi: f64) f64 {
    return @min(@max(x, -hi), hi);
}

pub fn terminal(s: [4]f64) bool {
    return -libm.cos(s[0]) - libm.cos(s[1] + s[0]) > 1.0;
}

/// One step for a single environment: the new state and whether it is terminal.
pub fn dynamics(s: [4]f64, action: Action) struct { s: [4]f64, terminated: bool } {
    const a = torques[action];
    const k1 = dsdt(s, a, lib(s));
    var y: [4]f64 = undefined;
    for (0..4) |i| y[i] = s[i] + dt2 * k1[i];
    const k2 = dsdt(y, a, lib(y));
    for (0..4) |i| y[i] = s[i] + dt2 * k2[i];
    const k3 = dsdt(y, a, lib(y));
    for (0..4) |i| y[i] = s[i] + dt * k3[i];
    const k4 = dsdt(y, a, lib(y));
    var n: [4]f64 = undefined;
    for (0..4) |i| n[i] = s[i] + dt6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    n[0] = wrap(n[0]);
    n[1] = wrap(n[1]);
    n[2] = bound(n[2], max_vel_1);
    n[3] = bound(n[3], max_vel_2);
    return .{ .s = n, .terminated = terminal(n) };
}

pub fn observe(s: [4]f64) [obs_dim]f32 {
    return .{ @floatCast(libm.cos(s[0])), @floatCast(libm.sin(s[0])), @floatCast(libm.cos(s[1])), @floatCast(libm.sin(s[1])), @floatCast(s[2]), @floatCast(s[3]) };
}

/// Gymnasium's reset distribution: each variable uniform on [-0.1, 0.1),
/// then rounded to float32.
pub fn initial(rand: std.Random) [4]f64 {
    var s: [4]f64 = undefined;
    for (&s) |*v| v.* = @as(f32, @floatCast(-0.1 + 0.2 * rand.float(f64)));
    return s;
}

pub const Batch = struct {
    n: usize,
    s: [4][]f64,
    y: [4][]f64, // the input of the current Runge-Kutta stage
    k: [4][4][]f64, // k1..k4, each with four components
    t: [8][]f64, // the fields of Lib, in order
    a: []f64,
    steps: []u32,

    pub fn init(gpa: std.mem.Allocator, n: usize) !Batch {
        const p = batch.pad(n);
        var b: Batch = .{ .n = n, .s = undefined, .y = undefined, .k = undefined, .t = undefined, .a = try gpa.alloc(f64, p), .steps = try gpa.alloc(u32, n) };
        for (&b.s) |*x| x.* = try gpa.alloc(f64, p);
        for (&b.y) |*x| x.* = try gpa.alloc(f64, p);
        for (&b.t) |*x| x.* = try gpa.alloc(f64, p);
        for (&b.k) |*kk| for (kk) |*x| {
            x.* = try gpa.alloc(f64, p);
        };
        for (b.s) |x| @memset(x, 0);
        for (b.y) |x| @memset(x, 0);
        for (b.t) |x| @memset(x, 0);
        for (b.k) |kk| for (kk) |x| @memset(x, 0);
        @memset(b.a, 0);
        @memset(b.steps, 0);
        return b;
    }

    pub fn deinit(b: *Batch, gpa: std.mem.Allocator) void {
        for (b.s) |x| gpa.free(x);
        for (b.y) |x| gpa.free(x);
        for (b.t) |x| gpa.free(x);
        for (b.k) |kk| for (kk) |x| gpa.free(x);
        gpa.free(b.a);
        gpa.free(b.steps);
    }

    pub fn get(b: *const Batch, i: usize) [4]f64 {
        return .{ b.s[0][i], b.s[1][i], b.s[2][i], b.s[3][i] };
    }

    pub fn set(b: *Batch, i: usize, s: [4]f64) void {
        for (0..4) |c| b.s[c][i] = s[c];
        b.steps[i] = 0;
    }

    /// The derivative at the stage input `src` into `out`: the library calls
    /// one value at a time, then the arithmetic on 8 lanes.
    fn stage(b: *Batch, src: [4][]f64, out: [4][]f64) void {
        const n = b.n;
        for (0..n) |i| {
            const t = lib(.{ src[0][i], src[1][i], src[2][i], src[3][i] });
            const vals = [8]f64{ t.c2, t.s2, t.c12, t.c1, t.dth1_sq, t.dth2_sq, t.d2, t.d2_sq };
            for (vals, 0..) |v, j| b.t[j][i] = v;
        }
        const L = batch.lanes;
        const sp = batch.splat;
        var i: usize = 0;
        while (i < n) : (i += L) {
            const dth1 = batch.load(src[2], i);
            const dth2 = batch.load(src[3], i);
            const c2 = batch.load(b.t[0], i);
            const s2 = batch.load(b.t[1], i);
            const d2 = batch.load(b.t[6], i);
            const a = batch.load(b.a, i);
            const d1 = sp(d1_a) + sp(m2) * (sp(d1_b) + sp(d1_c) * c2) + sp(inertia) + sp(inertia);
            const phi2 = sp(phi2_c) * batch.load(b.t[2], i);
            const phi1 = sp(phi1_a) * batch.load(b.t[5], i) * s2 - sp(phi1_b) * dth2 * dth1 * s2 + sp(phi1_c) * batch.load(b.t[3], i) + phi2;
            const ddth2 = (a + d2 / d1 * phi1 - sp(dd2_a) * batch.load(b.t[4], i) * s2 - phi2) / (sp(dd2_den) - batch.load(b.t[7], i) / d1);
            const ddth1 = -(d2 * ddth2 + phi1) / d1;
            batch.store(out[0], i, dth1);
            batch.store(out[1], i, dth2);
            batch.store(out[2], i, ddth1);
            batch.store(out[3], i, ddth2);
        }
    }

    /// y = s + h * k, componentwise.
    fn euler(b: *Batch, h: f64, k: [4][]f64) void {
        const L = batch.lanes;
        for (0..4) |c| {
            var i: usize = 0;
            while (i < b.n) : (i += L) batch.store(b.y[c], i, batch.load(b.s[c], i) + batch.splat(h) * batch.load(k[c], i));
        }
    }

    pub fn advance(b: *Batch, actions: []const Action, reward: []f64, terminated: []bool) void {
        const n = b.n;
        for (0..n) |i| b.a[i] = torques[actions[i]];
        b.stage(b.s, b.k[0]);
        b.euler(dt2, b.k[0]);
        b.stage(b.y, b.k[1]);
        b.euler(dt2, b.k[1]);
        b.stage(b.y, b.k[2]);
        b.euler(dt, b.k[2]);
        b.stage(b.y, b.k[3]);
        const L = batch.lanes;
        const sp = batch.splat;
        for (0..4) |c| {
            var i: usize = 0;
            while (i < n) : (i += L) {
                const sum = batch.load(b.k[0][c], i) + sp(2.0) * batch.load(b.k[1][c], i) + sp(2.0) * batch.load(b.k[2][c], i) + batch.load(b.k[3][c], i);
                batch.store(b.s[c], i, batch.load(b.s[c], i) + sp(dt6) * sum);
            }
        }
        for (0..n) |i| {
            b.s[0][i] = wrap(b.s[0][i]);
            b.s[1][i] = wrap(b.s[1][i]);
            b.s[2][i] = bound(b.s[2][i], max_vel_1);
            b.s[3][i] = bound(b.s[3][i], max_vel_2);
            terminated[i] = terminal(b.get(i));
            reward[i] = if (terminated[i]) 0.0 else -1.0;
            b.steps[i] += 1;
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
    const n = 19;
    var b = try Batch.init(gpa, n);
    defer b.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(5);
    const r = prng.random();
    var acts: [n]u8 = undefined;
    var reward: [n]f64 = undefined;
    var term: [n]bool = undefined;
    for (0..n) |i| {
        b.set(i, .{ r.float(f64) * 6 - 3, r.float(f64) * 6 - 3, r.float(f64) * 20 - 10, r.float(f64) * 40 - 20 });
        acts[i] = r.uintLessThan(u8, 3);
    }
    var before: [n][4]f64 = undefined;
    for (0..n) |i| before[i] = b.get(i);
    for (0..3) |_| {
        b.advance(&acts, &reward, &term);
        for (0..n) |i| {
            const one = dynamics(before[i], acts[i]);
            const got = b.get(i);
            for (0..4) |c| try testing.expectEqual(one.s[c], got[c]);
            try testing.expectEqual(one.terminated, term[i]);
            before[i] = got;
        }
    }
}

test "energy-free hanging state stays put" {
    const r = dynamics(.{ 0, 0, 0, 0 }, 1);
    for (r.s) |v| try testing.expectApproxEqAbs(@as(f64, 0), v, 1e-12);
    try testing.expect(!r.terminated);
}
