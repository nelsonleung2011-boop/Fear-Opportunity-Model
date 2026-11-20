# Minimal private web server used by "Launch Dashboard.bat".
# No Python or internet required. Binds to 127.0.0.1 only (this computer).
param([int]$Port = 8765)

$root = (Get-Location).Path
$listener = New-Object System.Net.HttpListener
$listener.Prefixes.Add("http://127.0.0.1:$Port/")
$listener.Start()

$mime = @{
  '.html' = 'text/html; charset=utf-8'
  '.htm'  = 'text/html; charset=utf-8'
  '.js'   = 'application/javascript; charset=utf-8'
  '.css'  = 'text/css; charset=utf-8'
  '.json' = 'application/json; charset=utf-8'
  '.csv'  = 'text/csv; charset=utf-8'
  '.svg'  = 'image/svg+xml'
  '.png'  = 'image/png'
  '.jpg'  = 'image/jpeg'
  '.ico'  = 'image/x-icon'
}

try {
  while ($listener.IsListening) {
    $ctx = $listener.GetContext()
    try {
      $rel = [Uri]::UnescapeDataString($ctx.Request.Url.AbsolutePath.TrimStart('/'))
      if ([string]::IsNullOrWhiteSpace($rel)) { $rel = 'index.html' }
      $full = [System.IO.Path]::GetFullPath([System.IO.Path]::Combine($root, $rel))
      if (-not $full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        $ctx.Response.StatusCode = 403
      } elseif (-not [System.IO.File]::Exists($full)) {
        $ctx.Response.StatusCode = 404
      } else {
        $ext = [System.IO.Path]::GetExtension($full).ToLowerInvariant()
        if ($mime.ContainsKey($ext)) { $ctx.Response.ContentType = $mime[$ext] }
        else { $ctx.Response.ContentType = 'application/octet-stream' }
        $bytes = [System.IO.File]::ReadAllBytes($full)
        $ctx.Response.ContentLength64 = $bytes.LongLength
        $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length)
      }
    } catch {
      try { $ctx.Response.StatusCode = 500 } catch { }
    } finally {
      try { $ctx.Response.Close() } catch { }
    }
  }
} finally {
  $listener.Stop()
  $listener.Close()
}
