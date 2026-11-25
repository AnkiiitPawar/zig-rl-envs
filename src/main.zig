//! zrl: batched classic-control environments and PPO.
//!
//!   zrl replay --env cartpole --in seqs.bin --out result.bin
//!   zrl bench  --env cartpole [--envs 1024] [--steps 2000] [--seed 1]
//!   zrl train  --env cartpole [--seed 1] [--steps 500000] [--log log.csv] ...
//!
//! replay: steps sequences of actions from given starting states, for the
//! parity checks against Gymnasium. Input: u32 sequences, u32 steps; then per
//! sequence the starting state (float64 each) and the actions (u8, or float32
//! for pendulum). Output: per sequence and step, the observation (float32
//! each), the reward (float64) and the terminated flag (u8), all zeros after
//! the sequence's episode has ended.

const std = @import("std");
const rl = @import("rl");
const ppo = rl.ppo;

const usage =
    \\usage: zrl replay --env NAME --in FILE --out FILE
    \\       zrl bench  --env NAME [--envs 1024] [--steps 2000] [--seed 1]
    \\       zrl train  --env NAME [--seed 1] [--steps N] [--log FILE] [--json FILE]
    \\NAME is cartpole, pendulum or acrobot.
    \\
;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 2) fail(io, "missing command");
    const cmd = args[1];
    var opts = std.StringHashMap([]const u8).init(arena);
    var i: usize = 2;
    while (i < args.len) : (i += 2) {
        if (i + 1 >= args.len or !std.mem.startsWith(u8, args[i], "--")) fail(io, "options are --name value");
        try opts.put(args[i][2..], args[i + 1]);
    }
    const env = opts.get("env") orelse fail(io, "missing --env");
    if (std.mem.eql(u8, cmd, "replay")) {
        inline for (.{ "cartpole", "pendulum", "acrobot" }) |name| {
            if (std.mem.eql(u8, env, name)) return replay(@field(rl, name), gpa, io, opts);
        }
        fail(io, "unknown environment");
    } else if (std.mem.eql(u8, cmd, "bench")) {
        inline for (.{ "cartpole", "pendulum", "acrobot" }) |name| {
            if (std.mem.eql(u8, env, name)) return bench(@field(rl, name), gpa, io, opts);
        }
        fail(io, "unknown environment");
    } else if (std.mem.eql(u8, cmd, "train")) {
        return ppo.trainCommand(gpa, io, env, opts) catch |e| {
            if (e == error.BadOption) fail(io, "bad option");
            return e;
        };
    } else fail(io, "unknown command");
}

fn intOpt(io: std.Io, opts: std.StringHashMap([]const u8), name: []const u8, default: u64) u64 {
    const v = opts.get(name) orelse return default;
    return std.fmt.parseInt(u64, v, 10) catch fail(io, "bad integer option");
}

fn replay(comptime E: type, gpa: std.mem.Allocator, io: std.Io, opts: std.StringHashMap([]const u8)) !void {
    const in_path = opts.get("in") orelse fail(io, "missing --in");
    const out_path = opts.get("out") orelse fail(io, "missing --out");
    const data = std.Io.Dir.cwd().readFileAlloc(io, in_path, gpa, .unlimited) catch fail(io, "cannot read --in");
    defer gpa.free(data);
    var r = std.Io.Reader.fixed(data);
    const nseq = try r.takeInt(u32, .little);
    const nsteps = try r.takeInt(u32, .little);
    var b = try E.Batch.init(gpa, nseq);
    defer b.deinit(gpa);
    const acts = try gpa.alloc(E.Action, @as(usize, nseq) * nsteps);
    defer gpa.free(acts);
    for (0..nseq) |s| {
        var st: [E.state_dim]f64 = undefined;
        for (&st) |*v| v.* = @bitCast(try r.takeInt(u64, .little));
        b.set(s, st);
        for (0..nsteps) |t| {
            acts[t * nseq + s] = if (E.Action == f32) @bitCast(try r.takeInt(u32, .little)) else try r.takeByte();
        }
    }
    const rec = E.obs_dim * 4 + 8 + 1;
    const out = try gpa.alloc(u8, @as(usize, nseq) * nsteps * rec);
    defer gpa.free(out);
    @memset(out, 0);
    const reward = try gpa.alloc(f64, nseq);
    defer gpa.free(reward);
    const term = try gpa.alloc(bool, nseq);
    defer gpa.free(term);
    const done = try gpa.alloc(bool, nseq);
    defer gpa.free(done);
    @memset(done, false);
    var o: [E.obs_dim]f32 = undefined;
    for (0..nsteps) |t| {
        b.advance(acts[t * nseq ..][0..nseq], reward, term);
        for (0..nseq) |s| {
            if (done[s]) continue;
            var w = std.Io.Writer.fixed(out[(s * nsteps + t) * rec ..][0..rec]);
            b.observeInto(s, &o);
            for (o) |v| try w.writeInt(u32, @bitCast(v), .little);
            try w.writeInt(u64, @bitCast(reward[s]), .little);
            try w.writeByte(@intFromBool(term[s]));
            if (term[s]) done[s] = true;
        }
    }
    const file = try std.Io.Dir.cwd().createFile(io, out_path, .{});
    defer file.close(io);
    var buf: [64 * 1024]u8 = undefined;
    var fw = file.writerStreaming(io, &buf);
    try fw.interface.writeAll(out);
    try fw.interface.flush();
}

/// Steps per second of the vector environment under random actions,
/// resets included.
fn bench(comptime E: type, gpa: std.mem.Allocator, io: std.Io, opts: std.StringHashMap([]const u8)) !void {
    const n: usize = @intCast(intOpt(io, opts, "envs", 1024));
    const steps: usize = @intCast(intOpt(io, opts, "steps", 2000));
    const seed = intOpt(io, opts, "seed", 1);
    var env = try rl.batch.VecEnv(E).init(gpa, n, seed);
    defer env.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(seed + 1);
    const r = prng.random();
    // The actions for every step are drawn before the clock starts.
    const acts = try gpa.alloc(E.Action, n * steps);
    defer gpa.free(acts);
    for (acts) |*a| a.* = if (E.Action == f32) (r.float(f32) * 4 - 2) else r.uintLessThan(u8, if (@hasDecl(E, "n_actions")) E.n_actions else 2);
    const t0 = std.Io.Timestamp.now(io, .awake);
    var checksum: f64 = 0;
    for (0..steps) |t| {
        env.step(acts[t * n ..][0..n]);
        checksum += env.reward[t % n];
    }
    const secs = @as(f64, @floatFromInt(t0.untilNow(io, .awake).nanoseconds)) / 1e9;
    var buf: [256]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buf);
    try stdout.interface.print("{{\"envs\": {d}, \"steps\": {d}, \"seconds\": {d}, \"steps_per_second\": {d}, \"episodes\": {d}, \"checksum\": {d}}}\n", .{ n, steps, secs, @as(f64, @floatFromInt(n * steps)) / secs, env.stats.count, checksum });
    try stdout.interface.flush();
}

fn fail(io: std.Io, msg: []const u8) noreturn {
    var buf: [1024]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(io, &buf);
    stderr.interface.print("zrl: {s}\n{s}", .{ msg, usage }) catch {};
    stderr.interface.flush() catch {};
    std.process.exit(1);
}
