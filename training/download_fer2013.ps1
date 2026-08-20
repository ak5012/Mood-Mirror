# Download FER2013 from Kaggle
# Requires: kaggle.json in ~/.kaggle/
$scriptDir = Split-Path -Parent $MyInvocation.MyCommandPath
Push-Location "$scriptDir\.."
New-Item -ItemType Directory -Path "data" -Force | Out-Null
Push-Location "data"
Write-Output "Downloading FER2013 from Kaggle..."
& kaggle datasets download -d msambare/fer2013
Write-Output "Extracting..."
Expand-Archive -Path "fer2013.zip" -DestinationPath "." -Force
Remove-Item -Path "fer2013.zip" -Force
Pop-Location
Write-Output "✓ FER2013 ready at: $(Get-Item fer2013.csv)"
Pop-Location
