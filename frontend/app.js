const trainingData = [
  { epoch: 1, loss: 2.84, accuracy: 0.22 },
  { epoch: 2, loss: 2.38, accuracy: 0.31 },
  { epoch: 3, loss: 2.02, accuracy: 0.39 },
  { epoch: 4, loss: 1.74, accuracy: 0.46 },
  { epoch: 5, loss: 1.52, accuracy: 0.53 },
  { epoch: 6, loss: 1.32, accuracy: 0.58 },
  { epoch: 7, loss: 1.15, accuracy: 0.63 },
  { epoch: 8, loss: 1.01, accuracy: 0.68 },
  { epoch: 9, loss: 0.9, accuracy: 0.72 },
  { epoch: 10, loss: 0.79, accuracy: 0.75 },
  { epoch: 11, loss: 0.69, accuracy: 0.78 },
  { epoch: 12, loss: 0.62, accuracy: 0.81 },
  { epoch: 13, loss: 0.55, accuracy: 0.83 },
  { epoch: 14, loss: 0.49, accuracy: 0.85 },
  { epoch: 15, loss: 0.44, accuracy: 0.86 },
  { epoch: 16, loss: 0.4, accuracy: 0.875 },
  { epoch: 17, loss: 0.36, accuracy: 0.886 },
  { epoch: 18, loss: 0.33, accuracy: 0.895 },
  { epoch: 19, loss: 0.31, accuracy: 0.901 },
  { epoch: 20, loss: 0.29, accuracy: 0.906 },
  { epoch: 21, loss: 0.27, accuracy: 0.91 },
  { epoch: 22, loss: 0.25, accuracy: 0.913 },
  { epoch: 23, loss: 0.24, accuracy: 0.916 },
  { epoch: 24, loss: 0.23, accuracy: 0.918 },
  { epoch: 25, loss: 0.22, accuracy: 0.919 },
  { epoch: 26, loss: 0.21, accuracy: 0.921 },
  { epoch: 27, loss: 0.2, accuracy: 0.922 },
  { epoch: 28, loss: 0.19, accuracy: 0.923 },
  { epoch: 29, loss: 0.185, accuracy: 0.922 },
  { epoch: 30, loss: 0.18, accuracy: 0.914 },
];

const chartCanvas = document.getElementById("trainingChart");
const networkCanvas = document.getElementById("networkCanvas");
const pulseButton = document.getElementById("pulseButton");

let pulseStart = performance.now();

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function drawTrainingChart() {
  const { ctx, width, height } = setupCanvas(chartCanvas);
  const padding = { top: 28, right: 62, bottom: 48, left: 58 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  drawGrid(ctx, padding, plotWidth, plotHeight);
  drawAxisLabels(ctx, width, height, padding);

  const maxLoss = 3;
  const pointX = (index) => padding.left + (index / (trainingData.length - 1)) * plotWidth;
  const lossY = (loss) => padding.top + (1 - loss / maxLoss) * plotHeight;
  const accuracyY = (accuracy) => padding.top + (1 - accuracy) * plotHeight;

  drawLine(ctx, trainingData.map((d, i) => [pointX(i), lossY(d.loss)]), "#d34f36", 3);
  drawLine(ctx, trainingData.map((d, i) => [pointX(i), accuracyY(d.accuracy)]), "#3858d6", 3);

  drawPoints(ctx, trainingData.map((d, i) => [pointX(i), lossY(d.loss)]), "#d34f36");
  drawPoints(ctx, trainingData.map((d, i) => [pointX(i), accuracyY(d.accuracy)]), "#3858d6");

  drawChartLabels(ctx, padding, plotWidth, plotHeight);
}

function drawGrid(ctx, padding, plotWidth, plotHeight) {
  ctx.strokeStyle = "#d8dfd9";
  ctx.lineWidth = 1;
  ctx.font = "12px Inter, system-ui, sans-serif";
  ctx.fillStyle = "#66736d";

  for (let i = 0; i <= 5; i += 1) {
    const y = padding.top + (i / 5) * plotHeight;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + plotWidth, y);
    ctx.stroke();
    ctx.fillText((3 - i * 0.6).toFixed(1), 18, y + 4);
    ctx.fillText(`${Math.round((1 - i / 5) * 100)}%`, padding.left + plotWidth + 18, y + 4);
  }

  for (let i = 0; i <= 6; i += 1) {
    const x = padding.left + (i / 6) * plotWidth;
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, padding.top + plotHeight);
    ctx.stroke();
    const epoch = Math.round(1 + (i / 6) * 29);
    ctx.fillText(String(epoch), x - 6, padding.top + plotHeight + 26);
  }
}

function drawAxisLabels(ctx, width, height, padding) {
  ctx.fillStyle = "#1d2522";
  ctx.font = "700 12px Inter, system-ui, sans-serif";
  ctx.fillText("Loss", 18, padding.top - 10);
  ctx.fillText("Accuracy", width - 116, padding.top - 10);
  ctx.fillText("Epoch", width / 2 - 18, height - 14);
}

function drawLine(ctx, points, color, lineWidth) {
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();
  points.forEach(([x, y], index) => {
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function drawPoints(ctx, points, color) {
  ctx.fillStyle = color;
  points.forEach(([x, y], index) => {
    if (index % 3 !== 0 && index !== points.length - 1) return;
    ctx.beginPath();
    ctx.arc(x, y, 3.7, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawChartLabels(ctx, padding, plotWidth, plotHeight) {
  const last = trainingData[trainingData.length - 1];
  ctx.fillStyle = "#1d2522";
  ctx.font = "700 12px Inter, system-ui, sans-serif";
  ctx.fillText(`Loss ${last.loss.toFixed(2)}`, padding.left + plotWidth - 88, padding.top + plotHeight * 0.34);
  ctx.fillText(
    `Accuracy ${(last.accuracy * 100).toFixed(1)}%`,
    padding.left + plotWidth - 120,
    padding.top + plotHeight * 0.12,
  );
}

function drawNetwork() {
  const { ctx, width, height } = setupCanvas(networkCanvas);
  const elapsed = performance.now() - pulseStart;
  const pulse = (elapsed % 2200) / 2200;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  const layers = [
    { label: "Input Tokens", count: 5 },
    { label: "Embedding", count: 7 },
    { label: "Encoder", count: 8 },
    { label: "Attention", count: 6 },
    { label: "Decoder", count: 8 },
    { label: "Output Tokens", count: 5 },
  ];

  const left = 58;
  const right = width - 58;
  const top = 78;
  const bottom = height - 70;
  const layerGap = (right - left) / (layers.length - 1);
  const nodes = layers.map((layer, layerIndex) => {
    const x = left + layerIndex * layerGap;
    const usableHeight = bottom - top;
    return Array.from({ length: layer.count }, (_, nodeIndex) => {
      const y = top + ((nodeIndex + 1) / (layer.count + 1)) * usableHeight;
      return { x, y, layerIndex, nodeIndex };
    });
  });

  drawConnections(ctx, nodes, pulse);
  drawNodes(ctx, nodes, pulse);
  drawLayerLabels(ctx, layers, left, layerGap, bottom);

  requestAnimationFrame(drawNetwork);
}

function drawConnections(ctx, nodes, pulse) {
  for (let layerIndex = 0; layerIndex < nodes.length - 1; layerIndex += 1) {
    const current = nodes[layerIndex];
    const next = nodes[layerIndex + 1];
    current.forEach((a, aIndex) => {
      next.forEach((b, bIndex) => {
        if ((aIndex + bIndex + layerIndex) % 2 !== 0) return;
        const phase = Math.abs(pulse - layerIndex / (nodes.length - 1));
        const intensity = Math.max(0.12, 0.85 - phase * 2.4);
        ctx.strokeStyle = `rgba(0, 124, 116, ${intensity})`;
        ctx.lineWidth = 0.8 + intensity * 1.6;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.bezierCurveTo(a.x + 42, a.y, b.x - 42, b.y, b.x, b.y);
        ctx.stroke();
      });
    });
  }
}

function drawNodes(ctx, nodes, pulse) {
  nodes.flat().forEach((node) => {
    const phase = Math.abs(pulse - node.layerIndex / 5);
    const active = Math.max(0, 1 - phase * 4);
    const radius = 7 + active * 5;

    ctx.fillStyle = active > 0.35 ? "#007c74" : "#e8f2ef";
    ctx.strokeStyle = active > 0.35 ? "#005b55" : "#9fb0a8";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  });
}

function drawLayerLabels(ctx, layers, left, layerGap, bottom) {
  ctx.fillStyle = "#1d2522";
  ctx.font = "700 12px Inter, system-ui, sans-serif";
  ctx.textAlign = "center";
  layers.forEach((layer, index) => {
    ctx.fillText(layer.label, left + index * layerGap, bottom + 34);
  });
  ctx.textAlign = "start";
}

function redraw() {
  drawTrainingChart();
  drawNetwork();
}

window.addEventListener("resize", drawTrainingChart);
pulseButton.addEventListener("click", () => {
  pulseStart = performance.now();
});

redraw();

