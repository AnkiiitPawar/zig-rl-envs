//! Proximal policy optimisation (Schulman et al., 2017) with generalised
//! advantage estimation (Schulman et al., 2016).
//!
//! The implementation choices follow CleanRL's ppo.py and
//! ppo_continuous_action.py, as catalogued in "The 37 implementation details
//! of proximal policy optimization" (Huang et al., 2022): separate actor and
//! critic networks with orthogonal initialisation, a state-independent log
//! standard deviation for continuous actions, advantages normalised per
//! minibatch, a clipped value loss, one Adam optimizer with epsilon 1e-5,
//! global gradient clipping, and a linearly annealed learning rate. One
//! difference: when an episode is cut off by the time limit rather than
//! ended, its last reward gets gamma times the critic's value of its last
//! observation added, instead of being treated as the end of the task.

const std = @import("std");
const batch = @import("batch.zig");
const nn = @import("nn.zig");
const normalize = @import("normalize.zig");
const cartpole = @import("cartpole.zig");
const pendulum = @import("pendulum.zig");
const acrobot = @import("acrobot.zig");
const Allocator = std.mem.Allocator;

pub const Config = struct {
    seed: u64 = 1,
    total_steps: usize = 500_000,
    num_envs: usize = 4,
    num_steps: usize = 128,
    lr: f32 = 2.5e-4,
    anneal_lr: bool = true,
    gamma: f32 = 0.99,
    gae_lambda: f32 = 0.95,
    num_minibatches: usize = 4,
    update_epochs: usize = 4,
    clip_coef: f32 = 0.2,
    clip_vloss: bool = true,
    ent_coef: f32 = 0.01,
    vf_coef: f32 = 0.5,
    max_grad_norm: f32 = 0.5,
    normalize_obs: bool = false,
    normalize_reward: bool = false,
    /// The environment counts as solved when the mean return of the last
    /// 100 training episodes reaches this.
    target: f64 = std.math.inf(f64),
    /// Stop as soon as the environment is solved.
    until_solved: bool = false,
    eval_episodes: usize = 100,
};

/// Settings per environment. CartPole and Acrobot use CleanRL's ppo.py
/// defaults. Pendulum uses the RL Baselines3 Zoo's PPO settings for it
/// (without state-dependent exploration), with CleanRL's observation and
/// reward normalisation.
pub fn defaults(env: []const u8) ?Config {
    if (std.mem.eql(u8, env, "cartpole")) return .{ .target = 475 };
    if (std.mem.eql(u8, env, "acrobot")) return .{ .target = -100 };
    if (std.mem.eql(u8, env, "pendulum")) return .{
        .total_steps = 500_000,
        .num_envs = 4,
        .num_steps = 1024,
        .lr = 1e-3,
        .gamma = 0.9,
        .num_minibatches = 64,
        .update_epochs = 10,
        .ent_coef = 0.0,
        .normalize_obs = true,
        .normalize_reward = true,
        .target = -200,
    };
    return null;
}

pub const Result = struct {
    steps: usize,
    seconds: f64,
    episodes: usize,
    solved_step: ?usize,
    solved_seconds: ?f64,
    final_mean_return: f64,
    eval_mean_return: f64,
};

const half_log_2pi: f32 = 0.5 * @log(2.0 * std.math.pi);

pub fn Trainer(comptime E: type) type {
    const discrete = E.Action == u8;
    const n_out = if (discrete) E.n_actions else E.act_dim;
    const od = E.obs_dim;
    return struct {
        const Self = @This();

        gpa: Allocator,
        cfg: Config,
        actor: nn.Mlp,
        critic: nn.Mlp,
        log_std: [n_out]f32 = @splat(0),
        log_std_grad: [n_out]f32 = @splat(0),
        adam: nn.Adam,
        obs_rms: normalize.RunningMeanStd,
        cache_a: nn.Mlp.Cache,
        cache_c: nn.Mlp.Cache,
        scratch: [2 * nn.hidden]f32 = undefined,
        prng: std.Random.DefaultPrng,

        pub fn init(gpa: Allocator, cfg: Config) !Self {
            var prng = std.Random.DefaultPrng.init(cfg.seed);
            const actor = try nn.Mlp.init(gpa, od, n_out);
            const critic = try nn.Mlp.init(gpa, od, 1);
            actor.initOrthogonal(prng.random(), 0.01);
            critic.initOrthogonal(prng.random(), 1.0);
            const cap = @max(cfg.num_envs * cfg.num_steps / cfg.num_minibatches, @max(cfg.num_envs, 16));
            var self: Self = .{
                .gpa = gpa,
                .cfg = cfg,
                .actor = actor,
                .critic = critic,
                .adam = undefined,
                .obs_rms = try normalize.RunningMeanStd.init(gpa, od),
                .cache_a = try nn.Mlp.Cache.init(gpa, cap),
                .cache_c = try nn.Mlp.Cache.init(gpa, cap),
                .prng = prng,
            };
            self.adam = try nn.Adam.init(gpa, &self.paramSlices());
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.actor.deinit(self.gpa);
            self.critic.deinit(self.gpa);
            self.adam.deinit(self.gpa);
            self.obs_rms.deinit(self.gpa);
            self.cache_a.deinit(self.gpa);
            self.cache_c.deinit(self.gpa);
        }

        fn paramSlices(self: *Self) [3][]f32 {
            return .{ self.actor.params, self.critic.params, &self.log_std };
        }

        fn gradSlices(self: *Self) [3][]f32 {
            return .{ self.actor.grads, self.critic.grads, &self.log_std_grad };
        }

        /// Copies raw observations and normalises them if enabled.
        fn prepare(self: *const Self, raw: []const f32, rows: usize, out: []f32) void {
            @memcpy(out[0 .. rows * od], raw[0 .. rows * od]);
            if (self.cfg.normalize_obs) self.obs_rms.apply(out, rows, 10);
        }

        fn logProb(self: *const Self, out: []const f32, act_d: u8, act_c: []const f32) f32 {
            if (discrete) {
                const lse = logSumExp(out);
                return out[act_d] - lse;
            } else {
                var lp: f32 = 0;
                for (0..n_out) |d| {
                    const sd = @exp(self.log_std[d]);
                    const z = (act_c[d] - out[d]) / sd;
                    lp += -0.5 * z * z - self.log_std[d] - half_log_2pi;
                }
                return lp;
            }
        }

        /// Trains and returns the result; logs one line per update to `log`.
        pub fn train(self: *Self, io: std.Io, log: ?*std.Io.Writer) !Result {
            const cfg = self.cfg;
            const gpa = self.gpa;
            const N = cfg.num_envs;
            const T = cfg.num_steps;
            const B = N * T;
            const MB = B / cfg.num_minibatches;
            const num_updates = cfg.total_steps / B;

            var env = try batch.VecEnv(E).init(gpa, N, cfg.seed);
            defer env.deinit(gpa);
            var scaler = try normalize.RewardScaler.init(gpa, N, cfg.gamma);
            defer scaler.deinit(gpa);

            const obs_buf = try gpa.alloc(f32, B * od);
            defer gpa.free(obs_buf);
            const act_d = try gpa.alloc(u8, B);
            defer gpa.free(act_d);
            const act_c = try gpa.alloc(f32, B * n_out);
            defer gpa.free(act_c);
            const logp_buf = try gpa.alloc(f32, B);
            defer gpa.free(logp_buf);
            const rew = try gpa.alloc(f32, B);
            defer gpa.free(rew);
            const done = try gpa.alloc(bool, B);
            defer gpa.free(done);
            const val = try gpa.alloc(f32, B);
            defer gpa.free(val);
            const adv = try gpa.alloc(f32, B);
            defer gpa.free(adv);
            const ret = try gpa.alloc(f32, B);
            defer gpa.free(ret);
            const perm = try gpa.alloc(usize, B);
            defer gpa.free(perm);
            for (perm, 0..) |*p, i| p.* = i;

            const cur = try gpa.alloc(f32, N * od);
            defer gpa.free(cur);
            const out_a = try gpa.alloc(f32, @max(N, MB) * n_out);
            defer gpa.free(out_a);
            const out_c = try gpa.alloc(f32, @max(N, MB));
            defer gpa.free(out_c);
            const d_a = try gpa.alloc(f32, MB * n_out);
            defer gpa.free(d_a);
            const d_c = try gpa.alloc(f32, MB);
            defer gpa.free(d_c);
            const mb_obs = try gpa.alloc(f32, MB * od);
            defer gpa.free(mb_obs);
            const env_act = try gpa.alloc(E.Action, N);
            defer gpa.free(env_act);
            const step_rew = try gpa.alloc(f32, N);
            defer gpa.free(step_rew);
            const step_done = try gpa.alloc(bool, N);
            defer gpa.free(step_done);
            const tmp = try gpa.alloc(f32, N);
            defer gpa.free(tmp);
            var one_obs: [od]f32 = undefined;
            var one_out: [1]f32 = undefined;

            const r = self.prng.random();
            const t0 = std.Io.Timestamp.now(io, .awake);
            var global_step: usize = 0;
            var solved_step: ?usize = null;
            var solved_seconds: ?f64 = null;
            var seen_episodes: usize = 0;

            if (cfg.normalize_obs) self.obs_rms.update(env.obs, N);
            self.prepare(env.obs, N, cur);

            var update: usize = 1;
            outer: while (update <= num_updates) : (update += 1) {
                const lr = if (cfg.anneal_lr) cfg.lr * (1 - @as(f32, @floatFromInt(update - 1)) / @as(f32, @floatFromInt(num_updates))) else cfg.lr;
                // Collect a rollout.
                for (0..T) |t| {
                    const base = t * N;
                    @memcpy(obs_buf[base * od ..][0 .. N * od], cur);
                    self.actor.forward(cur, N, self.cache_a, out_a);
                    self.critic.forward(cur, N, self.cache_c, out_c);
                    for (0..N) |i| {
                        const o = out_a[i * n_out ..][0..n_out];
                        const k = base + i;
                        val[k] = out_c[i];
                        if (discrete) {
                            const a = sampleCategorical(r, o);
                            act_d[k] = a;
                            env_act[i] = a;
                            logp_buf[k] = self.logProb(o, a, &.{});
                        } else {
                            const a = act_c[k * n_out ..][0..n_out];
                            for (0..n_out) |d| a[d] = o[d] + @exp(self.log_std[d]) * r.floatNorm(f32);
                            logp_buf[k] = self.logProb(o, 0, a);
                            env_act[i] = std.math.clamp(a[0], -E.max_torque, E.max_torque);
                        }
                    }
                    env.step(env_act);
                    global_step += N;
                    for (0..N) |i| {
                        step_rew[i] = @floatCast(env.reward[i]);
                        step_done[i] = env.terminated[i] or env.truncated[i];
                    }
                    if (cfg.normalize_reward) scaler.scale(step_rew, step_done, 10, tmp);
                    // Bootstrap episodes cut off by the time limit.
                    for (0..N) |i| {
                        if (env.truncated[i] and !env.terminated[i]) {
                            self.prepare(env.final_obs[i * od ..][0..od], 1, &one_obs);
                            self.critic.forward(&one_obs, 1, self.cache_c, &one_out);
                            step_rew[i] += cfg.gamma * one_out[0];
                        }
                        rew[base + i] = step_rew[i];
                        done[base + i] = step_done[i];
                    }
                    if (cfg.normalize_obs) self.obs_rms.update(env.obs, N);
                    self.prepare(env.obs, N, cur);
                    if (env.stats.count != seen_episodes) {
                        seen_episodes = env.stats.count;
                        if (solved_step == null and env.stats.count >= 100 and env.stats.meanReturn() >= cfg.target) {
                            solved_step = global_step;
                            solved_seconds = elapsed(io, t0);
                            if (cfg.until_solved) break :outer;
                        }
                    }
                }
                // Generalised advantage estimates.
                self.critic.forward(cur, N, self.cache_c, out_c);
                var gae: [256]f32 = @splat(0);
                std.debug.assert(N <= gae.len);
                var t: usize = T;
                while (t > 0) {
                    t -= 1;
                    for (0..N) |i| {
                        const k = t * N + i;
                        const next_v = if (t == T - 1) out_c[i] else val[k + N];
                        const nonterminal: f32 = if (done[k]) 0 else 1;
                        const delta = rew[k] + cfg.gamma * next_v * nonterminal - val[k];
                        gae[i] = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae[i];
                        adv[k] = gae[i];
                        ret[k] = gae[i] + val[k];
                    }
                }
                // Optimise.
                var stats: UpdateStats = .{};
                for (0..cfg.update_epochs) |_| {
                    r.shuffle(usize, perm);
                    var start: usize = 0;
                    while (start + MB <= B) : (start += MB) {
                        const idx = perm[start..][0..MB];
                        stats = self.minibatch(idx, obs_buf, act_d, act_c, logp_buf, val, adv, ret, mb_obs, out_a, out_c, d_a, d_c, lr);
                    }
                }
                if (log) |w| {
                    try w.print("{d},{d:.3},{d:.3},{d},{d:.4},{d:.4},{d:.4},{d:.4}\n", .{ global_step, elapsed(io, t0), env.stats.meanReturn(), env.stats.count, stats.pg_loss, stats.v_loss, stats.entropy, stats.approx_kl });
                }
            }
            const seconds = elapsed(io, t0);
            const eval_return = try self.evaluate(cfg.eval_episodes);
            return .{
                .steps = global_step,
                .seconds = seconds,
                .episodes = env.stats.count,
                .solved_step = solved_step,
                .solved_seconds = solved_seconds,
                .final_mean_return = env.stats.meanReturn(),
                .eval_mean_return = eval_return,
            };
        }

        const UpdateStats = struct { pg_loss: f32 = 0, v_loss: f32 = 0, entropy: f32 = 0, approx_kl: f32 = 0 };

        /// One gradient step on the samples `idx`.
        fn minibatch(self: *Self, idx: []const usize, obs_buf: []const f32, act_d: []const u8, act_c: []const f32, logp_buf: []const f32, val: []const f32, adv: []const f32, ret: []const f32, mb_obs: []f32, out_a: []f32, out_c: []f32, d_a: []f32, d_c: []f32, lr: f32) UpdateStats {
            const cfg = self.cfg;
            const M = idx.len;
            const mf: f32 = @floatFromInt(M);
            for (idx, 0..) |k, j| @memcpy(mb_obs[j * od ..][0..od], obs_buf[k * od ..][0..od]);
            self.actor.forward(mb_obs, M, self.cache_a, out_a);
            self.critic.forward(mb_obs, M, self.cache_c, out_c);
            // Normalised advantages (unbiased standard deviation, as torch.std).
            var mean: f32 = 0;
            for (idx) |k| mean += adv[k];
            mean /= mf;
            var sq: f32 = 0;
            for (idx) |k| sq += (adv[k] - mean) * (adv[k] - mean);
            const sd = @sqrt(sq / (mf - 1));
            var st: UpdateStats = .{};
            @memset(&self.log_std_grad, 0);
            for (idx, 0..) |k, j| {
                const o = out_a[j * n_out ..][0..n_out];
                const a_norm = (adv[k] - mean) / (sd + 1e-8);
                const ac = if (discrete) &[_]f32{} else act_c[k * n_out ..][0..n_out];
                const ad: u8 = if (discrete) act_d[k] else 0;
                const lp = self.logProb(o, ad, ac);
                const log_ratio = lp - logp_buf[k];
                const ratio = @exp(log_ratio);
                const pg1 = -a_norm * ratio;
                const pg2 = -a_norm * std.math.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef);
                st.pg_loss += @max(pg1, pg2) / mf;
                st.approx_kl += ((ratio - 1) - log_ratio) / mf;
                // d(loss)/d(log prob) for this sample.
                const dlp = (if (pg1 >= pg2) -a_norm else 0) * ratio / mf;
                const da = d_a[j * n_out ..][0..n_out];
                if (discrete) {
                    const lse = logSumExp(o);
                    var ent: f32 = 0;
                    for (o) |z| {
                        const p = @exp(z - lse);
                        ent -= p * (z - lse);
                    }
                    st.entropy += ent / mf;
                    for (o, da, 0..) |z, *g, c| {
                        const p = @exp(z - lse);
                        const onehot: f32 = if (c == ad) 1 else 0;
                        g.* = dlp * (onehot - p) + cfg.ent_coef / mf * p * ((z - lse) + ent);
                    }
                } else {
                    for (0..n_out) |d| {
                        const var_ = @exp(2 * self.log_std[d]);
                        const diff = ac[d] - o[d];
                        da[d] = dlp * diff / var_;
                        self.log_std_grad[d] += dlp * (diff * diff / var_ - 1);
                    }
                    for (0..n_out) |d| st.entropy += (0.5 + half_log_2pi + self.log_std[d]) / mf;
                }
                // Value loss.
                const v = out_c[j];
                const unclipped = (v - ret[k]) * (v - ret[k]);
                var dv = v - ret[k];
                var vl = 0.5 * unclipped;
                if (cfg.clip_vloss) {
                    const vc = val[k] + std.math.clamp(v - val[k], -cfg.clip_coef, cfg.clip_coef);
                    const clipped = (vc - ret[k]) * (vc - ret[k]);
                    if (clipped > unclipped) {
                        vl = 0.5 * clipped;
                        dv = if (@abs(v - val[k]) < cfg.clip_coef) vc - ret[k] else 0;
                    }
                }
                st.v_loss += vl / mf;
                d_c[j] = cfg.vf_coef * dv / mf;
            }
            if (!discrete) for (&self.log_std_grad) |*g| {
                g.* -= cfg.ent_coef;
            };
            @memset(self.actor.grads, 0);
            @memset(self.critic.grads, 0);
            self.actor.backward(mb_obs, M, self.cache_a, d_a, &self.scratch);
            self.critic.backward(mb_obs, M, self.cache_c, d_c, &self.scratch);
            var grads = self.gradSlices();
            const nslices: usize = if (discrete) 2 else 3;
            _ = nn.clipGradNorm(grads[0..nslices], cfg.max_grad_norm);
            var params = self.paramSlices();
            self.adam.step(params[0..nslices], grads[0..nslices], lr);
            return st;
        }

        /// Mean return of the deterministic policy (most likely action, or the
        /// mean) over `episodes` episodes in fresh environments.
        pub fn evaluate(self: *Self, episodes: usize) !f64 {
            if (episodes == 0) return std.math.nan(f64);
            const n: usize = @min(episodes, 16);
            var env = try batch.VecEnv(E).init(self.gpa, n, self.cfg.seed +% 1_000_003);
            defer env.deinit(self.gpa);
            const cur = try self.gpa.alloc(f32, n * od);
            defer self.gpa.free(cur);
            const out = try self.gpa.alloc(f32, n * n_out);
            defer self.gpa.free(out);
            const acts = try self.gpa.alloc(E.Action, n);
            defer self.gpa.free(acts);
            var total: f64 = 0;
            var counted: usize = 0;
            while (counted < episodes) {
                self.prepare(env.obs, n, cur);
                self.actor.forward(cur, n, self.cache_a, out);
                for (0..n) |i| {
                    const o = out[i * n_out ..][0..n_out];
                    acts[i] = if (discrete) @intCast(std.mem.indexOfMax(f32, o)) else std.math.clamp(o[0], -E.max_torque, E.max_torque);
                }
                const before = env.stats.count;
                env.step(acts);
                var k = before;
                while (k < env.stats.count and counted < episodes) : (k += 1) {
                    total += env.stats.returns[k % normalizeCap];
                    counted += 1;
                }
            }
            return total / @as(f64, @floatFromInt(episodes));
        }
    };
}

const normalizeCap = 100; // batch.Stats keeps the last 100 episodes

fn logSumExp(z: []const f32) f32 {
    var m = z[0];
    for (z[1..]) |v| m = @max(m, v);
    var s: f32 = 0;
    for (z) |v| s += @exp(v - m);
    return m + @log(s);
}

fn sampleCategorical(r: std.Random, logits: []const f32) u8 {
    const lse = logSumExp(logits);
    var u = r.float(f32);
    for (logits, 0..) |z, i| {
        u -= @exp(z - lse);
        if (u < 0) return @intCast(i);
    }
    return @intCast(logits.len - 1);
}

fn elapsed(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.untilNow(io, .awake).nanoseconds)) / 1e9;
}

/// `zrl train`: options override the environment's defaults.
pub fn trainCommand(gpa: Allocator, io: std.Io, env: []const u8, opts: std.StringHashMap([]const u8)) !void {
    var cfg = defaults(env) orelse return error.BadOption;
    var it = opts.iterator();
    while (it.next()) |kv| {
        const k = kv.key_ptr.*;
        const v = kv.value_ptr.*;
        if (std.mem.eql(u8, k, "env") or std.mem.eql(u8, k, "log") or std.mem.eql(u8, k, "json")) continue;
        const info = @typeInfo(Config).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (std.mem.eql(u8, k, name)) {
                @field(cfg, name) = switch (T) {
                    usize, u64 => std.fmt.parseInt(T, v, 10) catch return error.BadOption,
                    f32, f64 => std.fmt.parseFloat(T, v) catch return error.BadOption,
                    bool => std.mem.eql(u8, v, "1") or std.mem.eql(u8, v, "true"),
                    else => @compileError("unsupported option type"),
                };
                break;
            }
        } else return error.BadOption;
    }
    if (cfg.num_envs * cfg.num_steps % cfg.num_minibatches != 0) return error.BadOption;

    var log_file: ?std.Io.File = null;
    var log_buf: [64 * 1024]u8 = undefined;
    var log_writer: std.Io.File.Writer = undefined;
    if (opts.get("log")) |p| {
        log_file = try std.Io.Dir.cwd().createFile(io, p, .{});
        log_writer = log_file.?.writerStreaming(io, &log_buf);
        try log_writer.interface.writeAll("step,seconds,mean_return_100,episodes,pg_loss,v_loss,entropy,approx_kl\n");
    }
    defer if (log_file) |f| f.close(io);

    const res = blk: {
        if (std.mem.eql(u8, env, "cartpole")) {
            var tr = try Trainer(cartpole).init(gpa, cfg);
            defer tr.deinit();
            break :blk try tr.train(io, if (log_file != null) &log_writer.interface else null);
        } else if (std.mem.eql(u8, env, "acrobot")) {
            var tr = try Trainer(acrobot).init(gpa, cfg);
            defer tr.deinit();
            break :blk try tr.train(io, if (log_file != null) &log_writer.interface else null);
        } else {
            var tr = try Trainer(pendulum).init(gpa, cfg);
            defer tr.deinit();
            break :blk try tr.train(io, if (log_file != null) &log_writer.interface else null);
        }
    };
    if (log_file != null) try log_writer.interface.flush();

    const solved_step = if (res.solved_step) |v| try std.fmt.allocPrint(gpa, "{d}", .{v}) else try gpa.dupe(u8, "null");
    defer gpa.free(solved_step);
    const solved_seconds = if (res.solved_seconds) |v| try std.fmt.allocPrint(gpa, "{d}", .{v}) else try gpa.dupe(u8, "null");
    defer gpa.free(solved_seconds);
    const text = try std.fmt.allocPrint(gpa, "{{\"env\": \"{s}\", \"seed\": {d}, \"steps\": {d}, \"seconds\": {d}, \"episodes\": {d}, \"solved_step\": {s}, \"solved_seconds\": {s}, \"final_mean_return\": {d}, \"eval_mean_return\": {d}}}\n", .{ env, cfg.seed, res.steps, res.seconds, res.episodes, solved_step, solved_seconds, res.final_mean_return, res.eval_mean_return });
    defer gpa.free(text);
    var buf: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    try stdout.interface.writeAll(text);
    try stdout.interface.flush();
    if (opts.get("json")) |p| {
        const f = try std.Io.Dir.cwd().createFile(io, p, .{});
        defer f.close(io);
        var fb: [1024]u8 = undefined;
        var fw = f.writerStreaming(io, &fb);
        try fw.interface.writeAll(text);
        try fw.interface.flush();
    }
}
