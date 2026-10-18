# expose-wsl.ps1 -- let phones on the same Wi-Fi reach services running inside WSL2.
#
# WHY THIS IS NEEDED
#   Your voice assistant runs inside WSL2, which sits behind its own NAT
#   (172.22.48.1). Windows forwards localhost:8090 to WSL, but that only works
#   ON this machine. A phone hitting http://192.168.3.205:8090 reaches Windows,
#   which has nothing listening -> connection refused.
#
# USAGE (must be Administrator)
#   powershell -ExecutionPolicy Bypass -File windows\expose-wsl.ps1
#   powershell -ExecutionPolicy Bypass -File windows\expose-wsl.ps1 -Ports 8090,8080
#   powershell -ExecutionPolicy Bypass -File windows\expose-wsl.ps1 -Remove
#
# RE-RUN AFTER EVERY `wsl --shutdown`: the WSL IP changes on each boot.
#
# PREFERRED ALTERNATIVE
#   WSL 2.0+ on Win11 supports mirrored networking -- no portproxy needed:
#     add "networkingMode=mirrored" under [wsl2] in C:\Users\<you>\.wslconfig
#     then: wsl --shutdown
#
# NOTE: this only makes the page reachable. The MICROPHONE still needs https
#   (or an Android WebView shell). See docs/P1-voice.md.

param(
    [int[]]$Ports = @(8090),
    [switch]$Remove
)

$ErrorActionPreference = "Stop"

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Write-Host "[ERROR] Run this in an Administrator PowerShell." -ForegroundColor Red
    exit 1
}

if ($Remove) {
    foreach ($port in $Ports) {
        netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$port 2>$null
        Write-Host "[removed] port $port" -ForegroundColor Yellow
    }
    Write-Host "`nCurrent portproxy rules:"
    netsh interface portproxy show v4tov4
    exit 0
}

# --- find the WSL IP (first IPv4 of the default distro) ---
$raw = & wsl.exe -e hostname -I 2>$null
if (-not $raw) {
    Write-Host "[ERROR] Cannot get WSL IP. Is WSL running?" -ForegroundColor Red
    exit 1
}
$wslIp = ($raw -split '\s+')[0]
Write-Host "[WSL] IP = $wslIp"
if ($wslIp -notmatch '^\d+\.\d+\.\d+\.\d+$') {
    Write-Host "[ERROR] Unexpected WSL IP: $wslIp" -ForegroundColor Red
    exit 1
}

# --- local LAN IP (the one the phone should use) ---
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 |
          Where-Object { $_.IPAddress -notlike '127.*' -and
                         $_.IPAddress -notlike '169.254.*' -and
                         $_.IPAddress -notlike '172.2*' -and
                         $_.IPAddress -notlike '192.168.235.*' } |
          Sort-Object InterfaceMetric | Select-Object -First 1).IPAddress

foreach ($port in $Ports) {
    netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$port 2>$null | Out-Null
    netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$port `
        connectaddress=$wslIp connectport=$port | Out-Null
    Write-Host "[proxy] 0.0.0.0:$port -> ${wslIp}:$port" -ForegroundColor Green
}

# --- firewall (idempotent) ---
$ruleName = "EV voice assistant (WSL)"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow `
        -Protocol TCP -LocalPort $Ports | Out-Null
    Write-Host "[firewall] rule created for ports $($Ports -join ',')" -ForegroundColor Green
} else {
    Write-Host "[firewall] rule already exists"
}

Write-Host ""
Write-Host "Phone (same Wi-Fi) open: http://${lanIp}:$($Ports[0])" -ForegroundColor Cyan
Write-Host "Reminder: page loads there, but the MIC still needs https." -ForegroundColor Yellow
Write-Host "Verify from this PC first: curl http://${lanIp}:$($Ports[0])/api/status"
