# 3D Point Cloud


![Vista del proyecto](image.png)

## Uso

Reconstrucción del video de prueba (`test/0x01.mp4`, 200 frames 1920x1080), genera `clouds/map.ply` y `clouds/mesh.ply`:

```bash
.venv/bin/python main.py ./test/0x01.mp4 --fov 60 --voxel-size 0.01 --sdf-trunc 0.1 --min-weight 3 --min-component 0.01 --smooth-iterations 10 --step 1 --out clouds
```

Parámetros de calidad de la malla:

- `--min-weight`: frames que deben ver una superficie para extraerla (más alto = menos ruido, más huecos).
- `--sdf-trunc`: truncado del TSDF; conviene ~10x `--voxel-size` para promediar el ruido de la profundidad monocular.
- `--min-component`: borra islas sueltas con menos de esa fracción de triángulos (`0` desactiva).
- `--smooth-iterations`: suavizado Taubin de la malla (`0` desactiva).
