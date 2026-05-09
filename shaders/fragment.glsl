uniform sampler2D uAtlas;
uniform float uAtlasCols;
uniform float uAtlasRows;
varying float vTexIndex;

void main() {
  float idx = floor(vTexIndex + 0.5);
  float col = mod(idx, uAtlasCols);
  float row = floor(idx / uAtlasCols);
  vec2 cellSize = vec2(1.0 / uAtlasCols, 1.0 / uAtlasRows);
  vec2 uv = vec2(col, row) * cellSize + gl_PointCoord * cellSize;
  uv.y = 1.0 - uv.y;
  vec4 color = texture2D(uAtlas, uv);
  if (color.a < 0.01) discard;
  gl_FragColor = color;
}
