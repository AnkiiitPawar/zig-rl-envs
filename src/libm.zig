//! The C library's sin, cos, fmod and pow.
//!
//! Gymnasium's classic-control environments compute with NumPy, whose float64
//! sin and cos return the same bits as the C library's (glibc's libm;
//! scripts/parity.py checks this on every run). A NumPy scalar's `x**2` is
//! pow(x, 2), which differs from x * x in the last bit for about 1 input in
//! 1,000; on a float32 scalar it is powf. Calling the same functions here is
//! what makes the trajectories bit-identical.
//!
//! They are called through glibc's sinf64, cosf64, fmodf64, powf64 and
//! powf32, which glibc exports as aliases of sin, cos, fmod, pow and powf.
//! Plain `sin` would bind to the compiler's built-in version, which differs
//! from glibc's in the last bit for some inputs. On other C libraries Zig's
//! own functions are used, and the results are no longer bit-identical to
//! Gymnasium's.

const std = @import("std");
const builtin = @import("builtin");

const glibc = builtin.target.isGnuLibC();

const c = struct {
    extern "m" fn sinf64(x: f64) f64;
    extern "m" fn cosf64(x: f64) f64;
    extern "m" fn fmodf64(x: f64, y: f64) f64;
    extern "m" fn powf64(x: f64, y: f64) f64;
    extern "m" fn powf32(x: f32, y: f32) f32;
};

pub fn sin(x: f64) f64 {
    return if (glibc) c.sinf64(x) else @sin(x);
}

pub fn cos(x: f64) f64 {
    return if (glibc) c.cosf64(x) else @cos(x);
}

pub fn fmod(x: f64, y: f64) f64 {
    return if (glibc) c.fmodf64(x, y) else @rem(x, y);
}

/// NumPy's `x**2` for a float64 scalar.
pub fn sq(x: f64) f64 {
    return if (glibc) c.powf64(x, 2.0) else x * x;
}

/// NumPy's `x**2` for a float32 scalar.
pub fn sqf(x: f32) f32 {
    return if (glibc) c.powf32(x, 2.0) else x * x;
}

/// Python's and NumPy's float remainder: the result has the sign of `b`.
pub fn pyMod(a: f64, b: f64) f64 {
    var r = fmod(a, b);
    if (r != 0) {
        if ((b < 0) != (r < 0)) r += b;
    } else r = if (b < 0) -0.0 else 0.0;
    return r;
}
