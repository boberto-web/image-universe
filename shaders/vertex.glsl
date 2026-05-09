attribute float aSize;
attribute float aTexIndex;
varying float vTexIndex;

void main() {
  vTexIndex = aTexIndex;
  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = aSize * (400.0 / -mvPosition.z);
  gl_Position = projectionMatrix * mvPosition;
}
