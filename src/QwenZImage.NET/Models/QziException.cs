namespace QwenZImage.NET;

/// <summary>A failure reported by the native engine or the ABI guard.</summary>
public sealed class QziException : Exception
{
    /// <summary>Native status code.</summary>
    public QziStatus Status { get; }

    /// <summary>Verbatim message written by the native library.</summary>
    public string NativeMessage { get; }

    /// <summary>Creates an exception from a native status and message.</summary>
    public QziException(QziStatus status, string message)
        : base($"qzi error {(int)status} ({status}): {message}")
    {
        Status = status;
        NativeMessage = message;
    }

    /// <summary>Creates an exception with an inner exception.</summary>
    public QziException(QziStatus status, string message, Exception inner)
        : base($"qzi error {(int)status} ({status}): {message}", inner)
    {
        Status = status;
        NativeMessage = message;
    }
}
