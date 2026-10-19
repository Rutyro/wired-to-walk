# MaleCNS connectome data

The graph loader uses the public **MaleCNS v1.0** release from the Janelia
Research Campus:

- Download page: <https://male-cns.janelia.org/download/>
- Tables: `body-annotations`, `body-neurotransmitters` and
  `connectome-weights` from
  `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`
- License: CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>)

The tables are not committed to this repository. Fetch them with:

```powershell
pwsh ./scripts/fetch-malecns.ps1
```

Attribution is repeated in `THIRD_PARTY_NOTICES.md` and the README. If you
publish derived graphs or figures, keep the MaleCNS attribution and the CC BY
4.0 notice with them.
