#!/bin/bash

echo "⚙️ Creating Scanner Wrapper..."

# Gera o script do wrapper no /tmp usando um heredoc limpo
cat << 'EOF' > /tmp/scan_wrapper.sh
#!/bin/bash
ARGS=("$@")
FILTERED_ARGS=()
for arg in "${ARGS[@]}"; do
  if [[ "$arg" != "--acpp-targets=generic" ]]; then
    FILTERED_ARGS+=("$arg")
  fi
done
exec /usr/bin/clang-scan-deps "${FILTERED_ARGS[@]}"
EOF

chmod +x /tmp/scan_wrapper.sh

echo "⚙️ Initializing CMake..."
cmake --preset=docker-sycl -DACPP_TARGETS=generic &&
cmake --build  --preset=docker-sycl &&
      /app/out/build/docker-sycl/dimna_sycl/di_mna_sycl \
      --in /app/${DATASET_INPUT} \
      --out /app/output/ \
      --runs ${NUM_RUNS} \
      --cs ${PAR_ALFA} \
      --ccn ${PAR_BETA};


echo "🔭 Watching for changes on /app/src..."

# Loop de monitoramento (inotifywait)
while inotifywait -q -r -e modify,create,delete /app/src --exclude 'nohup.out|output|run.py|build|out'; do
  echo "🔄 Change detected! Recompiling..."
  cmake --build --preset=docker-sycl &&
      /app/out/build/docker-sycl/dimna_sycl/di_mna_sycl \
      --in /app/${DATASET_INPUT} \
      --out /app/output/ \
      --runs "${NUM_RUNS}" \
      --cs "${PAR_ALFA}" \
      --ccn "${PAR_BETA}";
  echo "✅ Execution finished. Waiting for changes..."
done
