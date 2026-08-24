use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

const CMAKE_FLAGS: &[&str] = &[
    "-DNRD_STATIC_LIBRARY=ON",
    "-DNRD_EMBEDS_SPIRV_SHADERS=ON",
    "-DNRD_EMBEDS_DXIL_SHADERS=OFF",
    "-DNRD_EMBEDS_DXBC_SHADERS=OFF",
    "-DNRD_NORMAL_ENCODING=0",    // RGBA8_UNORM world-space normals
    "-DNRD_ROUGHNESS_ENCODING=1", // linear roughness
    "-DSHADERMAKE_FIND_DXC=OFF",  // DXC comes from the Vulkan SDK
];

fn nrd_lib() -> PathBuf {
    Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap_or_default())
        .join("third_party/nrd/_Bin/Release/NRD.lib")
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=third_party/nrd/Include/NRD.h");

    let lib = nrd_lib();
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap_or_default();

    if !lib.exists() {
        let ok = Command::new("cmake")
            .args(CMAKE_FLAGS)
            .arg("-DFETCHCONTENT_SOURCE_DIR_SHADERMAKE")
            .arg(Path::new(&manifest).join("third_party/shadermake"))
            .arg("-DFETCHCONTENT_SOURCE_DIR_MATHLIB")
            .arg(Path::new(&manifest).join("third_party/mathlib"))
            .arg("-S")
            .arg(Path::new(&manifest).join("third_party/nrd"))
            .arg("-B")
            .arg(Path::new(&manifest).join("third_party/nrd/build"))
            .status()
            .is_ok_and(|s| s.success());

        assert!(ok, "cmake configure of third_party/nrd failed");

        let ok = Command::new("cmake")
            .args(["--build", "third_party/nrd/build", "--config", "Release"])
            .current_dir(&manifest)
            .status()
            .is_ok_and(|s| s.success());

        assert!(ok, "cmake build of third_party/nrd failed");
    }

    println!(
        "cargo:rustc-link-search=native={}",
        lib.parent().and_then(Path::to_str).unwrap_or_default()
    );
    println!("cargo:rustc-link-lib=static=NRD");

    // NRD's static library resolves shader-blob permutations through
    // ShaderMake's blob reader, which its CMake target links privately.
    let manifest_dir = Path::new(&manifest);
    println!(
        "cargo:rustc-link-search=native={}",
        manifest_dir
            .join("third_party/nrd/build/_deps/shadermake-build/Release")
            .to_str()
            .unwrap_or_default()
    );
    println!("cargo:rustc-link-lib=static=ShaderMakeBlob");
}
