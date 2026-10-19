# flybody model

The simulation uses the public flybody model from
<https://github.com/TuragaLab/flybody> (Apache-2.0).

The model and its meshes are not committed here because the full asset set is
large. Fetch them with:

```powershell
pwsh ./scripts/fetch-flybody.ps1
```

The script places `fruitfly.xml`, `floor.xml`, the mesh assets and the upstream
`LICENSE` in `data/flybody/`. The Rust simulator loads
`data/flybody/fruitfly.xml` by default.

Do not add a modified MJCF to the repository without updating
`THIRD_PARTY_NOTICES.md` and preserving the upstream license.
