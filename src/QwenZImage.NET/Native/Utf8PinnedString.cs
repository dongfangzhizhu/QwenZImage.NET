using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;

namespace QwenZImage.NET.Native;

/// <summary>
/// A NUL-terminated UTF-8 string pinned for the duration of a native call.
/// Buffers come from <see cref="ArrayPool{T}"/>, so a generation request does
/// not leave prompt-sized garbage on the GC heap. Strings are only read by the
/// engine during the call (ABI rule 3), so the pin can be released immediately
/// afterwards.
/// </summary>
internal sealed unsafe class Utf8PinnedString : IDisposable
{
    private byte[]? _buffer;
    private GCHandle _pin;

    public byte* Pointer { get; private set; }

    public Utf8PinnedString(string? value)
    {
        if (value is null) return;
        int count = Encoding.UTF8.GetByteCount(value);
        _buffer = ArrayPool<byte>.Shared.Rent(count + 1);
        int written = Encoding.UTF8.GetBytes(value.AsSpan(), _buffer.AsSpan());
        _buffer[written] = 0;
        _pin = GCHandle.Alloc(_buffer, GCHandleType.Pinned);
        Pointer = (byte*)_pin.AddrOfPinnedObject();
    }

    public void Dispose()
    {
        if (_pin.IsAllocated) _pin.Free();
        if (_buffer is not null)
        {
            ArrayPool<byte>.Shared.Return(_buffer);
            _buffer = null;
        }
        Pointer = null;
    }
}
