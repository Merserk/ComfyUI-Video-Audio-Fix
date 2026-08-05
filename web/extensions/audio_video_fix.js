import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";

const PREVIEW_STATE = Symbol("VideoAudioFix.videoPreview");
const DEFAULT_VIDEO_RATIO = 9 / 16;
const HORIZONTAL_PADDING = 4;

function mediaUrl(item) {
  const params = new URLSearchParams({
    filename: item?.filename || "",
    subfolder: item?.subfolder || "",
    type: item?.type || "output",
    rand: Math.random().toString(36).slice(2),
  });
  return api.apiURL(`/view?${params.toString()}`);
}

function isAudioVideoFixNode(node) {
  return (
    node?.type === "AudioVideoFix" ||
    node?.comfyClass === "AudioVideoFix" ||
    node?.constructor?.comfyClass === "AudioVideoFix"
  );
}

function getPreviewState(node) {
  return node?.[PREVIEW_STATE] || null;
}

function dirtyCanvas(node) {
  node.graph?.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function resizeNodeToContent(node) {
  const computed = node.computeSize?.();
  if (!computed || !node.setSize) return;

  const currentWidth = Number(node.size?.[0] || 0);
  const nextWidth = Math.max(currentWidth, Number(computed[0] || 0));
  const nextHeight = Math.max(120, Number(computed[1] || 0));
  node.setSize([nextWidth, nextHeight]);
}

function setWidgetHeight(node, state) {
  if (state.updatingHeight) return;
  state.updatingHeight = true;
  try {
    const availableWidth = Math.max(220, Number(node?.size?.[0] || 360) - 24 - HORIZONTAL_PADDING);
    const height = Math.ceil(availableWidth * (state.aspectRatio || DEFAULT_VIDEO_RATIO));
    state.height = height;
    const pixels = `${height}px`;
    state.container.style.height = pixels;
    state.container.style.minHeight = pixels;
    state.container.style.setProperty("--comfy-widget-height", pixels);
    state.container.style.setProperty("--comfy-widget-min-height", pixels);
    resizeNodeToContent(node);
    dirtyCanvas(node);
  } finally {
    state.updatingHeight = false;
  }
}

function hidePreview(node, state) {
  state.height = 0;
  state.aspectRatio = DEFAULT_VIDEO_RATIO;
  state.container.replaceChildren();
  state.container.style.display = "none";
  state.container.style.height = "0px";
  state.container.style.minHeight = "0px";
  state.container.style.setProperty("--comfy-widget-height", "0px");
  state.container.style.setProperty("--comfy-widget-min-height", "0px");
  state.widget.hidden = true;
  resizeNodeToContent(node);
  dirtyCanvas(node);
}

function appendLoadError(parent) {
  const error = document.createElement("div");
  error.textContent = "Video preview could not be loaded";
  error.style.padding = "10px 4px";
  error.style.fontSize = "11px";
  error.style.textAlign = "center";
  error.style.opacity = "0.75";
  parent.append(error);
}

function renderPreview(node, output) {
  const state = getPreviewState(node);
  if (!state) return;

  const videoItem = Array.isArray(output?.video_preview)
    ? output.video_preview.find((item) => item?.filename)
    : null;

  if (!videoItem) {
    hidePreview(node, state);
    return;
  }

  state.container.replaceChildren();
  state.widget.hidden = false;
  state.container.style.display = "block";

  const wrapper = document.createElement("div");
  wrapper.style.width = "100%";
  wrapper.style.margin = "0";
  wrapper.style.padding = "0";
  wrapper.style.overflow = "hidden";
  wrapper.style.lineHeight = "0";
  wrapper.style.display = "flex";
  wrapper.style.justifyContent = "center";

  const video = document.createElement("video");
  video.controls = true;
  video.preload = "metadata";
  video.playsInline = true;
  video.style.width = "100%";
  video.style.height = "auto";
  video.style.display = "block";
  video.style.margin = "0";
  video.style.padding = "0";
  video.style.objectFit = "contain";
  video.style.borderRadius = "3px";

  const settle = () => {
    if (video.videoWidth > 0 && video.videoHeight > 0) {
      state.aspectRatio = video.videoHeight / video.videoWidth;
      video.style.maxWidth = `${video.videoWidth}px`;
    } else {
      state.aspectRatio = DEFAULT_VIDEO_RATIO;
    }
    setWidgetHeight(node, state);
  };

  video.addEventListener("loadedmetadata", settle, { once: true });
  video.addEventListener(
    "error",
    () => {
      video.style.display = "none";
      appendLoadError(wrapper);
      state.aspectRatio = 0.2;
      setWidgetHeight(node, state);
    },
    { once: true },
  );

  wrapper.append(video);
  state.container.append(wrapper);
  video.src = mediaUrl(videoItem);

  if (video.readyState >= 1) queueMicrotask(settle);
  else setWidgetHeight(node, state);
}

function installPreviewWidget(node) {
  if (getPreviewState(node)) return;

  const container = document.createElement("div");
  container.style.display = "none";
  container.style.width = "100%";
  container.style.height = "0px";
  container.style.minHeight = "0px";
  container.style.margin = "0";
  container.style.padding = `0 ${HORIZONTAL_PADDING / 2}px`;
  container.style.boxSizing = "border-box";
  container.style.overflow = "hidden";

  const state = {
    container,
    widget: null,
    height: 0,
    aspectRatio: DEFAULT_VIDEO_RATIO,
    updatingHeight: false,
  };

  const widget = node.addDOMWidget("video-preview", "div", container, {
    hideOnZoom: false,
    getMinHeight: () => state.height,
    getHeight: () => state.height,
    afterResize: () => {
      if (state.height > 0) setWidgetHeight(node, state);
    },
  });
  widget.serialize = false;
  widget.options.serialize = false;
  widget.hidden = true;
  state.widget = widget;
  node[PREVIEW_STATE] = state;
}

function findRootGraphNode(nodeLocatorId) {
  const graph = app.graph;
  if (!graph) return null;

  const direct = graph.getNodeById?.(nodeLocatorId);
  if (direct) return direct;

  const numericId = Number(nodeLocatorId);
  if (Number.isFinite(numericId)) {
    const numeric = graph.getNodeById?.(numericId);
    if (numeric) return numeric;
  }

  return graph._nodes?.find((node) => String(node.id) === String(nodeLocatorId)) || null;
}

app.registerExtension({
  name: "VideoAudioFix.VideoPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AudioVideoFix") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      installPreviewWidget(this);
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      renderPreview(this, output);
    };
  },

  nodeCreated(node) {
    if (isAudioVideoFixNode(node)) installPreviewWidget(node);
  },

  onNodeOutputsUpdated(outputsByNodeId) {
    for (const [nodeLocatorId, output] of Object.entries(outputsByNodeId || {})) {
      const node = findRootGraphNode(nodeLocatorId);
      if (!isAudioVideoFixNode(node)) continue;
      installPreviewWidget(node);
      renderPreview(node, output);
    }
  },
});
