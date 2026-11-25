//! A small multilayer perceptron in float32: two hidden layers of 64 tanh
//! units and a linear output, with its gradient, orthogonal initialisation
//! and the Adam optimizer.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const hidden = 64;
const VL = 8;
const Vf = @Vector(VL, f32);

/// The weights and biases live in one array, laid out as
/// w1 (hidden x in), b1, w2 (hidden x hidden), b2, w3 (out x hidden), b3,
/// so that the optimizer and gradient clipping can treat them as one vector.
pub const Mlp = struct {
    in: usize,
    out: usize,
    params: []f32,
    grads: []f32,

    pub fn size(in: usize, out: usize) usize {
        return hidden * in + hidden + hidden * hidden + hidden + out * hidden + out;
    }

    pub fn init(gpa: Allocator, in: usize, out: usize) !Mlp {
        const n = size(in, out);
        const m: Mlp = .{ .in = in, .out = out, .params = try gpa.alloc(f32, n), .grads = try gpa.alloc(f32, n) };
        @memset(m.params, 0);
        @memset(m.grads, 0);
        return m;
    }

    pub fn deinit(m: Mlp, gpa: Allocator) void {
        gpa.free(m.params);
        gpa.free(m.grads);
    }

    const Parts = struct { w1: []f32, b1: []f32, w2: []f32, b2: []f32, w3: []f32, b3: []f32 };

    fn parts(m: Mlp, a: []f32) Parts {
        var o: usize = 0;
        const take = struct {
            fn f(s: []f32, off: *usize, len: usize) []f32 {
                defer off.* += len;
                return s[off.*..][0..len];
            }
        }.f;
        return .{
            .w1 = take(a, &o, hidden * m.in),
            .b1 = take(a, &o, hidden),
            .w2 = take(a, &o, hidden * hidden),
            .b2 = take(a, &o, hidden),
            .w3 = take(a, &o, m.out * hidden),
            .b3 = take(a, &o, m.out),
        };
    }

    /// Orthogonal weights scaled by sqrt(2) in the hidden layers and by
    /// `out_gain` in the last one, and zero biases, as in CleanRL's PPO.
    pub fn initOrthogonal(m: Mlp, rand: std.Random, out_gain: f32) void {
        const p = m.parts(m.params);
        orthogonal(rand, hidden, m.in, std.math.sqrt2, p.w1);
        orthogonal(rand, hidden, hidden, std.math.sqrt2, p.w2);
        orthogonal(rand, m.out, hidden, out_gain, p.w3);
        @memset(p.b1, 0);
        @memset(p.b2, 0);
        @memset(p.b3, 0);
    }

    /// Activations of one forward pass, kept for the backward pass.
    pub const Cache = struct {
        cap: usize,
        h1: []f32,
        h2: []f32,

        pub fn init(gpa: Allocator, cap: usize) !Cache {
            return .{ .cap = cap, .h1 = try gpa.alloc(f32, cap * hidden), .h2 = try gpa.alloc(f32, cap * hidden) };
        }

        pub fn deinit(c: Cache, gpa: Allocator) void {
            gpa.free(c.h1);
            gpa.free(c.h2);
        }
    };

    /// y = f(x) for a batch: x is rows x in, y is rows x out (row-major).
    pub fn forward(m: Mlp, x: []const f32, rows: usize, cache: Cache, y: []f32) void {
        std.debug.assert(rows <= cache.cap);
        const p = m.parts(m.params);
        for (0..rows) |r| {
            const xr = x[r * m.in ..][0..m.in];
            const h1 = cache.h1[r * hidden ..][0..hidden];
            for (0..hidden) |o| {
                var s = p.b1[o];
                for (p.w1[o * m.in ..][0..m.in], xr) |w, v| s += w * v;
                h1[o] = std.math.tanh(s);
            }
            const h2 = cache.h2[r * hidden ..][0..hidden];
            for (0..hidden) |o| h2[o] = std.math.tanh(p.b2[o] + dot(p.w2[o * hidden ..][0..hidden], h1));
            const yr = y[r * m.out ..][0..m.out];
            for (0..m.out) |o| yr[o] = p.b3[o] + dot(p.w3[o * hidden ..][0..hidden], h2);
        }
    }

    /// Adds the gradient of sum(dy . y) with respect to the parameters to
    /// m.grads, for the batch of the last forward pass. `scratch` holds
    /// 2 x hidden floats.
    pub fn backward(m: Mlp, x: []const f32, rows: usize, cache: Cache, dy: []const f32, scratch: []f32) void {
        const p = m.parts(m.params);
        const g = m.parts(m.grads);
        const d2 = scratch[0..hidden];
        const d1 = scratch[hidden..][0..hidden];
        for (0..rows) |r| {
            const h1 = cache.h1[r * hidden ..][0..hidden];
            const h2 = cache.h2[r * hidden ..][0..hidden];
            const dyr = dy[r * m.out ..][0..m.out];
            // Output layer.
            @memset(d2, 0);
            for (0..m.out) |o| {
                const d = dyr[o];
                if (d == 0) continue;
                g.b3[o] += d;
                axpy(g.w3[o * hidden ..][0..hidden], d, h2);
                axpy(d2, d, p.w3[o * hidden ..][0..hidden]);
            }
            // Second hidden layer.
            for (d2, h2) |*d, h| d.* *= 1 - h * h;
            @memset(d1, 0);
            for (0..hidden) |o| {
                const d = d2[o];
                g.b2[o] += d;
                axpy(g.w2[o * hidden ..][0..hidden], d, h1);
                axpy(d1, d, p.w2[o * hidden ..][0..hidden]);
            }
            // First hidden layer.
            for (d1, h1) |*d, h| d.* *= 1 - h * h;
            const xr = x[r * m.in ..][0..m.in];
            for (0..hidden) |o| {
                const d = d1[o];
                g.b1[o] += d;
                for (g.w1[o * m.in ..][0..m.in], xr) |*w, v| w.* += d * v;
            }
        }
    }
};

fn dot(a: []const f32, b: []const f32) f32 {
    var acc: Vf = @splat(0);
    var i: usize = 0;
    while (i + VL <= a.len) : (i += VL) acc += @as(Vf, a[i..][0..VL].*) * @as(Vf, b[i..][0..VL].*);
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) s += a[i] * b[i];
    return s;
}

/// y += a * x
fn axpy(y: []f32, a: f32, x: []const f32) void {
    const av: Vf = @splat(a);
    var i: usize = 0;
    while (i + VL <= y.len) : (i += VL) y[i..][0..VL].* = @as(Vf, y[i..][0..VL].*) + av * @as(Vf, x[i..][0..VL].*);
    while (i < y.len) : (i += 1) y[i] += a * x[i];
}

/// A rows x cols matrix with orthonormal rows (if rows <= cols) or columns,
/// times `gain`: Gram-Schmidt on a Gaussian matrix, with the signs fixed as
/// in a QR decomposition whose R has a positive diagonal.
pub fn orthogonal(rand: std.Random, rows: usize, cols: usize, gain: f32, out: []f32) void {
    const k = @min(rows, cols); // number of orthonormal vectors
    const n = @max(rows, cols); // their length
    var buf: [hidden * hidden]f64 = undefined;
    var vecs = buf[0 .. k * n];
    for (vecs) |*v| v.* = rand.floatNorm(f64);
    for (0..k) |i| {
        const vi = vecs[i * n ..][0..n];
        for (0..i) |j| {
            const vj = vecs[j * n ..][0..n];
            var d: f64 = 0;
            for (vi, vj) |a, b| d += a * b;
            for (vi, vj) |*a, b| a.* -= d * b;
        }
        var norm: f64 = 0;
        for (vi) |a| norm += a * a;
        norm = @sqrt(norm);
        for (vi) |*a| a.* /= norm;
    }
    for (0..rows) |r| for (0..cols) |c| {
        const v = if (rows <= cols) vecs[r * n + c] else vecs[c * n + r];
        out[r * cols + c] = @floatCast(gain * v);
    };
}

/// Adam (Kingma and Ba, 2015) over several parameter arrays, as PyTorch
/// implements it.
pub const Adam = struct {
    m: [][]f32,
    v: [][]f32,
    t: u32 = 0,
    beta1: f32 = 0.9,
    beta2: f32 = 0.999,
    eps: f32 = 1e-5,

    pub fn init(gpa: Allocator, shapes: []const []f32) !Adam {
        const m = try gpa.alloc([]f32, shapes.len);
        const v = try gpa.alloc([]f32, shapes.len);
        for (shapes, m, v) |s, *a, *b| {
            a.* = try gpa.alloc(f32, s.len);
            b.* = try gpa.alloc(f32, s.len);
            @memset(a.*, 0);
            @memset(b.*, 0);
        }
        return .{ .m = m, .v = v };
    }

    pub fn deinit(a: Adam, gpa: Allocator) void {
        for (a.m) |x| gpa.free(x);
        for (a.v) |x| gpa.free(x);
        gpa.free(a.m);
        gpa.free(a.v);
    }

    pub fn step(a: *Adam, params: []const []f32, grads: []const []f32, lr: f32) void {
        a.t += 1;
        const t: f32 = @floatFromInt(a.t);
        const bc1 = 1 - std.math.pow(f32, a.beta1, t);
        const bc2 = 1 - std.math.pow(f32, a.beta2, t);
        const step_size = lr / bc1;
        const sbc2 = @sqrt(bc2);
        for (params, grads, a.m[0..params.len], a.v[0..params.len]) |p, g, m, v| {
            for (p, g, m, v) |*pi, gi, *mi, *vi| {
                mi.* = a.beta1 * mi.* + (1 - a.beta1) * gi;
                vi.* = a.beta2 * vi.* + (1 - a.beta2) * gi * gi;
                pi.* -= step_size * mi.* / (@sqrt(vi.*) / sbc2 + a.eps);
            }
        }
    }
};

/// Scales the gradients so that their joint Euclidean norm is at most
/// `max_norm` (PyTorch's clip_grad_norm_); returns the norm before.
pub fn clipGradNorm(grads: []const []f32, max_norm: f32) f32 {
    var total: f64 = 0;
    for (grads) |g| for (g) |x| {
        total += @as(f64, x) * x;
    };
    const norm: f32 = @floatCast(@sqrt(total));
    const coef = max_norm / (norm + 1e-6);
    if (coef < 1) for (grads) |g| for (g) |*x| {
        x.* *= coef;
    };
    return norm;
}

const testing = std.testing;

test "the gradient matches finite differences" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    const m = try Mlp.init(gpa, 3, 2);
    defer m.deinit(gpa);
    m.initOrthogonal(r, 1.0);
    for (m.params) |*p| p.* += 0.1 * r.floatNorm(f32);
    const rows = 5;
    var x: [rows * 3]f32 = undefined;
    for (&x) |*v| v.* = r.floatNorm(f32);
    var dy: [rows * 2]f32 = undefined;
    for (&dy) |*v| v.* = r.floatNorm(f32);
    const cache = try Mlp.Cache.init(gpa, rows);
    defer cache.deinit(gpa);
    var y: [rows * 2]f32 = undefined;
    var scratch: [2 * hidden]f32 = undefined;
    m.forward(&x, rows, cache, &y);
    @memset(m.grads, 0);
    m.backward(&x, rows, cache, &dy, &scratch);
    // Check a sample of parameters against central differences of sum(dy . y).
    var checked: usize = 0;
    var i: usize = 0;
    while (i < m.params.len) : (i += 37) {
        const saved = m.params[i];
        const h: f32 = 1e-2;
        m.params[i] = saved + h;
        m.forward(&x, rows, cache, &y);
        var up: f64 = 0;
        for (y, dy) |a, b| up += a * b;
        m.params[i] = saved - h;
        m.forward(&x, rows, cache, &y);
        var down: f64 = 0;
        for (y, dy) |a, b| down += a * b;
        m.params[i] = saved;
        const numeric: f32 = @floatCast((up - down) / (2 * h));
        try testing.expectApproxEqAbs(numeric, m.grads[i], 2e-3 * @max(1, @abs(numeric)));
        checked += 1;
    }
    try testing.expect(checked > 100);
}

test "orthogonal matrices have orthonormal rows or columns" {
    var prng = std.Random.DefaultPrng.init(8);
    var w: [64 * 6]f32 = undefined;
    orthogonal(prng.random(), 64, 6, 1, &w); // tall: columns orthonormal
    for (0..6) |a| for (0..6) |b| {
        var d: f32 = 0;
        for (0..64) |r| d += w[r * 6 + a] * w[r * 6 + b];
        try testing.expectApproxEqAbs(@as(f32, if (a == b) 1 else 0), d, 1e-5);
    };
    var u: [2 * 64]f32 = undefined;
    orthogonal(prng.random(), 2, 64, 1, &u); // wide: rows orthonormal
    var d: f32 = 0;
    for (0..64) |c| d += u[c] * u[64 + c];
    try testing.expectApproxEqAbs(@as(f32, 0), d, 1e-5);
}

test "Adam minimises a quadratic" {
    const gpa = testing.allocator;
    var p = [_]f32{ 3, -2 };
    var g = [_]f32{ 0, 0 };
    const ps = [_][]f32{&p};
    const gs = [_][]f32{&g};
    var adam = try Adam.init(gpa, &ps);
    defer adam.deinit(gpa);
    for (0..2000) |_| {
        g[0] = 2 * (p[0] - 1);
        g[1] = 2 * (p[1] + 0.5);
        adam.step(&ps, &gs, 0.01);
    }
    try testing.expectApproxEqAbs(@as(f32, 1), p[0], 1e-3);
    try testing.expectApproxEqAbs(@as(f32, -0.5), p[1], 1e-3);
}
