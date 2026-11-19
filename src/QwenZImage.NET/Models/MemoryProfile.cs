namespace QwenZImage.NET;

/// <summary>
/// Low-VRAM budget preset. Mirrors the C ABI's <c>qzi_memory_profile</c> and
/// WanGP's mmgp profiles: a named bundle of allocator / VAE-encode levers so a
/// caller does not need to know the individual knobs.
/// </summary>
public enum QziMemoryProfile
{
    /// <summary>Resolve a concrete profile from the device's total VRAM.</summary>
    Auto = 0,

    /// <summary>Fastest: keep freed allocator blocks pooled (24 GiB threshold).</summary>
    High = 1,

    /// <summary>Balanced budget: moderate pool threshold, full encode cap.</summary>
    Balanced = 2,

    /// <summary>Low VRAM: tighter pool (1 GiB) and a 512 px reference encode cap.</summary>
    Low = 3,

    /// <summary>Lowest VRAM: tightest pool (256 MiB) and a 384 px encode cap.</summary>
    Lowest = 4
}

/// <summary>
/// Resolved, read-only low-VRAM budget for the current engine. VRAM totals are
/// device-wide (the numbers nvidia-smi reports); -1 means the device could not
/// be queried (e.g. the deterministic reference backend without a GPU).
/// </summary>
public readonly record struct QziMemoryPlan(
    QziMemoryProfile Profile,
    long TotalVramMib,
    long FreeVramMib,
    int PoolReleaseMib,
    int ReferenceEncodeCap);
