# Register (or remove) the daily Windows scheduled task for v7 operations.
# Usage:
#   powershell -ExecutionPolicy Bypass -File pipeline\register_daily_task.ps1
#   powershell -ExecutionPolicy Bypass -File pipeline\register_daily_task.ps1 -Time "08:30"
#   powershell -ExecutionPolicy Bypass -File pipeline\register_daily_task.ps1 -Unregister
# ASCII only on purpose: avoids encoding corruption in non-ASCII user paths.
param(
    [switch]$Unregister,
    [string]$Time = "08:30"
)

$ErrorActionPreference = "Stop"
$taskName = "ETF_Entry_Daily_v7"
$root = Split-Path -Parent $PSScriptRoot

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Removed scheduled task $taskName"
    return
}

$cmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $cmd) { throw "python was not found on PATH" }
$python = $cmd.Source

$action = New-ScheduledTaskAction -Execute $python -Argument "pipeline/ops_daily.py" -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
$settings.MultipleInstances = "IgnoreNew"

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Description "Daily ETF Entry Decision System refresh, score, rank and alert" -Force | Out-Null
Write-Host "Registered $taskName at $Time daily"
Write-Host "Working directory: $root"
Write-Host "Dry-run by default unless SMTP_HOST, SMTP_USER, SMTP_PASS and ALERT_TO are set."
