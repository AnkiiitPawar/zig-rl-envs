//! Shared pieces of the batched environments: vector helpers, and a vector
//! environment that resets finished episodes and keeps episode statistics.

const std = @import("std");

pub const lanes = 8;
pub const V = @Vector(lanes, f64);

/// Arrays are padded to a whole number of vectors.
pub fn pad(n: usize) usize {
    return (n + lanes - 1) / lanes * lanes;
}

pub inline fn load(a: []const f64, i: usize) V {
    return a[i..][0..lanes].*;
}

pub inline fn store(a: []f64, i: usize, v: V) void {
    a[i..][0..lanes].* = v;
}

pub inline fn splat(x: f64) V {
    return @splat(x);
}

/// Returns of recent finished episodes.
pub const Stats = struct {
    const cap = 100;
    returns: [cap]f64 = undefined,
    lengths: [cap]u32 = undefined,
    count: usize = 0,

    pub fn add(s: *Stats, ret: f64, len: u32) void {
        s.returns[s.count % cap] = ret;
        s.lengths[s.count % cap] = len;
        s.count += 1;
    }

    /// Mean return of the last `cap` episodes (or of all, if fewer).
    pub fn meanReturn(s: *const Stats) f64 {
        const k: usize = @min(s.count, cap);
        if (k == 0) return std.math.nan(f64);
        var t: f64 = 0;
        for (s.returns[0..k]) |r| t += r;
        return t / @as(f64, @floatFromInt(k));
    }
};

/// `n` copies of environment `E`, stepped together. When an episode ends
/// (terminated, or truncated at E.max_steps as Gymnasium's TimeLimit does),
/// its last observation goes to `final_obs` and the environment is reset in
/// the same step, so `obs` always holds observations to act on.
pub fn VecEnv(comptime E: type) type {
    return struct {
        const Self = @This();
        pub const Env = E;

        n: usize,
        b: E.Batch,
        prng: std.Random.DefaultPrng,
        obs: []f32,
        final_obs: []f32,
        reward: []f64,
        terminated: []bool,
        truncated: []bool,
        ep_return: []f64,
        stats: Stats = .{},

        pub fn init(gpa: std.mem.Allocator, n: usize, seed: u64) !Self {
            var self: Self = .{
                .n = n,
                .b = try E.Batch.init(gpa, n),
                .prng = std.Random.DefaultPrng.init(seed),
                .obs = try gpa.alloc(f32, n * E.obs_dim),
                .final_obs = try gpa.alloc(f32, n * E.obs_dim),
                .reward = try gpa.alloc(f64, n),
                .terminated = try gpa.alloc(bool, n),
                .truncated = try gpa.alloc(bool, n),
                .ep_return = try gpa.alloc(f64, n),
            };
            @memset(self.final_obs, 0);
            @memset(self.reward, 0);
            @memset(self.terminated, false);
            @memset(self.truncated, false);
            @memset(self.ep_return, 0);
            for (0..n) |i| self.resetOne(i);
            return self;
        }

        pub fn deinit(self: *Self, gpa: std.mem.Allocator) void {
            self.b.deinit(gpa);
            gpa.free(self.obs);
            gpa.free(self.final_obs);
            gpa.free(self.reward);
            gpa.free(self.terminated);
            gpa.free(self.truncated);
            gpa.free(self.ep_return);
        }

        fn resetOne(self: *Self, i: usize) void {
            self.b.set(i, E.initial(self.prng.random()));
            self.b.observeInto(i, self.obs[i * E.obs_dim ..][0..E.obs_dim]);
            self.ep_return[i] = 0;
        }

        pub fn step(self: *Self, actions: []const E.Action) void {
            self.b.advance(actions, self.reward, self.terminated);
            for (0..self.n) |i| {
                const o = self.obs[i * E.obs_dim ..][0..E.obs_dim];
                self.b.observeInto(i, o);
                self.truncated[i] = self.b.steps[i] >= E.max_steps;
                self.ep_return[i] += self.reward[i];
                if (self.terminated[i] or self.truncated[i]) {
                    @memcpy(self.final_obs[i * E.obs_dim ..][0..E.obs_dim], o);
                    self.stats.add(self.ep_return[i], self.b.steps[i]);
                    self.resetOne(i);
                }
            }
        }
    };
}
