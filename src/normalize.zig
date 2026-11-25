//! Running normalisation of observations and rewards, as Gymnasium's
//! NormalizeObservation and NormalizeReward wrappers do it.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Running mean and variance of a vector, updated a batch at a time
/// (Chan, Golub and LeVeque's parallel formula).
pub const RunningMeanStd = struct {
    mean: []f64,
    variance: []f64,
    count: f64 = 1e-4,

    pub fn init(gpa: Allocator, dim: usize) !RunningMeanStd {
        const r: RunningMeanStd = .{ .mean = try gpa.alloc(f64, dim), .variance = try gpa.alloc(f64, dim) };
        @memset(r.mean, 0);
        @memset(r.variance, 1);
        return r;
    }

    pub fn deinit(r: RunningMeanStd, gpa: Allocator) void {
        gpa.free(r.mean);
        gpa.free(r.variance);
    }

    /// Updates with `rows` vectors stored row-major in `x`.
    pub fn update(r: *RunningMeanStd, x: []const f32, rows: usize) void {
        const dim = r.mean.len;
        const n: f64 = @floatFromInt(rows);
        const total = r.count + n;
        for (0..dim) |d| {
            var bm: f64 = 0;
            for (0..rows) |i| bm += x[i * dim + d];
            bm /= n;
            var bv: f64 = 0;
            for (0..rows) |i| {
                const e = x[i * dim + d] - bm;
                bv += e * e;
            }
            bv /= n;
            const delta = bm - r.mean[d];
            const m2 = r.variance[d] * r.count + bv * n + delta * delta * r.count * n / total;
            r.mean[d] += delta * n / total;
            r.variance[d] = m2 / total;
        }
        r.count = total;
    }

    /// (x - mean) / sqrt(variance + 1e-8), clipped to [-clip, clip], in place.
    pub fn apply(r: *const RunningMeanStd, x: []f32, rows: usize, clip: f32) void {
        const dim = r.mean.len;
        for (0..rows) |i| for (0..dim) |d| {
            const z: f32 = @floatCast((x[i * dim + d] - r.mean[d]) / @sqrt(r.variance[d] + 1e-8));
            x[i * dim + d] = std.math.clamp(z, -clip, clip);
        };
    }
};

/// Divides rewards by the running standard deviation of each environment's
/// discounted return.
pub const RewardScaler = struct {
    returns: []f64,
    rms: RunningMeanStd,
    gamma: f64,

    pub fn init(gpa: Allocator, n: usize, gamma: f64) !RewardScaler {
        const s: RewardScaler = .{ .returns = try gpa.alloc(f64, n), .rms = try RunningMeanStd.init(gpa, 1), .gamma = gamma };
        @memset(s.returns, 0);
        return s;
    }

    pub fn deinit(s: RewardScaler, gpa: Allocator) void {
        gpa.free(s.returns);
        s.rms.deinit(gpa);
    }

    /// Scales `reward` in place (clipped to [-clip, clip]); `done` marks the
    /// environments whose episodes just ended.
    pub fn scale(s: *RewardScaler, reward: []f32, done: []const bool, clip: f32, scratch: []f32) void {
        for (s.returns, reward, scratch) |*ret, r, *out| {
            ret.* = ret.* * s.gamma + r;
            out.* = @floatCast(ret.*);
        }
        s.rms.update(scratch, reward.len);
        const sd: f32 = @floatCast(@sqrt(s.rms.variance[0] + 1e-8));
        for (reward, done, s.returns) |*r, d, *ret| {
            r.* = std.math.clamp(r.* / sd, -clip, clip);
            if (d) ret.* = 0;
        }
    }
};

const testing = std.testing;

test "batched updates give the variance of all the data" {
    const gpa = testing.allocator;
    var r = try RunningMeanStd.init(gpa, 1);
    defer r.deinit(gpa);
    r.count = 0;
    r.variance[0] = 0;
    var prng = std.Random.DefaultPrng.init(1);
    var all: [300]f32 = undefined;
    for (&all) |*v| v.* = 3 + 2 * prng.random().floatNorm(f32);
    var i: usize = 0;
    while (i < all.len) : (i += 30) r.update(all[i..][0..30], 30);
    var mean: f64 = 0;
    for (all) |v| mean += v;
    mean /= all.len;
    var variance: f64 = 0;
    for (all) |v| variance += (v - mean) * (v - mean);
    variance /= all.len;
    try testing.expectApproxEqRel(mean, r.mean[0], 1e-9);
    try testing.expectApproxEqRel(variance, r.variance[0], 1e-9);
}
