#!/bin/bash
set -e
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

cmake --preset=docker-sycl -DACPP_TARGETS=generic &&
cmake --build  --preset=docker-sycl

