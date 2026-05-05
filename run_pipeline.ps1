# ================================================================
# run_pipeline.ps1
# Executa todos os notebooks do pipeline em sequência.
# Uso: .\run_pipeline.ps1
#      .\run_pipeline.ps1 -Start 3          # começa no notebook 03
#      .\run_pipeline.ps1 -Start 3 -End 6   # roda só do 03 ao 06
# ================================================================
param(
    [int]$Start = 1,
    [int]$End   = 10
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$notebooks = @(
    @{ num=1;  name="01_ingest.ipynb";             timeout=600  },
    @{ num=2;  name="02_build_dataset_m2.ipynb";   timeout=900  },
    @{ num=3;  name="03_splits.ipynb";             timeout=900  },
    @{ num=4;  name="04_feature_selection.ipynb";  timeout=1200 },
    @{ num=5;  name="05_model.ipynb";              timeout=1200 },
    @{ num=6;  name="06_tuning.ipynb";             timeout=1800 },
    @{ num=7;  name="07_shap.ipynb";               timeout=1200 },
    @{ num=8;  name="08_score.ipynb";              timeout=600  },
    @{ num=9;  name="09_roi.ipynb";                timeout=600  },
    @{ num=10; name="10_monitor.ipynb";            timeout=600  }
)

$selected = $notebooks | Where-Object { $_.num -ge $Start -and $_.num -le $End }

$total   = $selected.Count
$current = 0
$failed  = @()

Write-Host ""
Write-Host "Pipeline: notebooks $Start → $End  ($total notebooks)" -ForegroundColor Cyan
Write-Host ("=" * 60)

foreach ($nb in $selected) {
    $current++
    $path = Join-Path $root "notebooks\$($nb.name)"
    Write-Host ""
    Write-Host "[$current/$total] $($nb.name)  (timeout=$($nb.timeout)s)" -ForegroundColor Yellow

    $t0 = Get-Date
    jupyter nbconvert --to notebook --execute --inplace `
        "--ExecutePreprocessor.timeout=$($nb.timeout)" `
        $path 2>&1
    $exit = $LASTEXITCODE
    $elapsed = [int]((Get-Date) - $t0).TotalSeconds

    if ($exit -eq 0) {
        Write-Host "  ✓ OK  (${elapsed}s)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ FALHOU  (exit=$exit, ${elapsed}s)" -ForegroundColor Red
        $failed += $nb.name
        # Continua os próximos notebooks mesmo com falha
    }
}

Write-Host ""
Write-Host ("=" * 60)
if ($failed.Count -eq 0) {
    Write-Host "Pipeline concluído com sucesso." -ForegroundColor Green
} else {
    Write-Host "Pipeline concluído com $($failed.Count) falha(s):" -ForegroundColor Red
    $failed | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
}
