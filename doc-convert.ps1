# 设置源目录和目标目录
$sourceRoot = "D:\data"
$targetRoot = "D:\converted_docs"

# 创建目标根目录（如果不存在）
if (!(Test-Path $targetRoot)) {
    New-Item -ItemType Directory -Path $targetRoot | Out-Null
}

# 遍历所有 .doc 文件（递归）
Get-ChildItem -Path $sourceRoot -Recurse -Filter *.doc | Where-Object { $_.Name -notlike '~$*' } | ForEach-Object {
    $sourceFile = $_.FullName
    # 计算相对路径
    $relativePath = $_.FullName.Substring($sourceRoot.Length).TrimStart('\')
    $targetDir = Join-Path $targetRoot ([System.IO.Path]::GetDirectoryName($relativePath))
    # 创建目标文件夹（如果不存在）
    if (!(Test-Path $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir | Out-Null
    }
    # 目标文件名
    $targetFile = Join-Path $targetDir ($_.BaseName + ".docx")
    if (Test-Path $targetFile) {
        Write-Host "Already exists: $targetFile"
    } else {
        Write-Host "Converting: $sourceFile -> $targetFile"
        # 用 soffice 转换
        & soffice --headless --convert-to docx --outdir $targetDir $sourceFile
        # 检查转换结果
        if (Test-Path $targetFile) {
            Write-Host "Success: $targetFile"
        } else {
            Write-Host "Convert failed: $sourceFile"
        }
    }
}

Write-Host "All conversions completed!"