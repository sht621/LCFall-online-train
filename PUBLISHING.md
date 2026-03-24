# Publishing `online_train`

This workspace is prepared so the online-side source code can be uploaded to GitHub without model weights.

## Included

- `src/online_train/camera/`
- `src/online_train/lidar/`
- `src/online_train/fusion_mlp/`
- `src/online_train/scripts/`
- `src/online_train/docs/`
- `src/online_train/manifests/lidar_roi_config.json`
- `src/online_train/manifests/split_config.json`
- `src/online_train/README.md`
- `src/online_train/COPY_SOURCES.md`

## Excluded

- `src/online_train/**/checkpoints/`
- `src/online_train/generated/`
- `src/online_train/annotations/`
- dataset-derived manifest files such as `camera_train.json`
- all `*.pth` files

## Notes

- The current source still contains absolute paths rooted at `/LCFall`. That is acceptable for archival sharing, but not ideal for reuse on another machine.
- If you want this repository to be directly runnable elsewhere, the next cleanup step is to replace absolute paths with environment-variable or project-root based resolution.
- If you only want to publish the online inference side, create a separate branch or subtree that keeps only the files listed above.
