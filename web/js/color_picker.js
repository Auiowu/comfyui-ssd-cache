import { app } from "../../scripts/app.js";

/**
 * 给「创建空白图片」节点加一个原生颜色选择器，
 * 选择后同步到 hex_color 文本框。失败时静默降级，不影响节点本身。
 */
app.registerExtension({
  name: "ComfyuiSSDCache.ColorPicker",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CreateBlankImage") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);
      try {
        attachColorPicker(this);
      } catch (err) {
        console.warn("[SSD-Cache] color picker attach failed:", err);
      }
    };
  },
});

function normalizeHex(value) {
  if (typeof value !== "string") return "#7f7f7f";
  let v = value.trim();
  if (!v.startsWith("#")) v = "#" + v;
  if (v.length === 4) {
    v = "#" + v[1] + v[1] + v[2] + v[2] + v[3] + v[3];
  }
  return /^#[0-9a-fA-F]{6}$/.test(v) ? v : "#7f7f7f";
}

function attachColorPicker(node) {
  const widget = node.widgets?.find((w) => w.name === "hex_color");
  if (!widget || widget._ssdColorPickerAttached) return;
  widget._ssdColorPickerAttached = true;

  const input = document.createElement("input");
  input.type = "color";
  input.value = normalizeHex(widget.value);
  input.style.cssText =
    "width:100%;height:100%;border:none;border-radius:4px;background:transparent;cursor:pointer;padding:0;";

  if (typeof node.addDOMWidget !== "function") {
    return;
  }

  node.addDOMWidget("ssd_color_pick", "color", input, {
    serialize: false,
    hideOnZoom: false,
    getValue: () => input.value,
    setValue: (v) => {
      input.value = normalizeHex(v);
    },
  });

  input.addEventListener("input", () => {
    widget.value = input.value;
    node.setDirtyCanvas?.(true, true);
  });

  // 用户在文本框里手改十六进制时，同步回选择器
  const originalCallback = widget.callback;
  widget.callback = function (value) {
    input.value = normalizeHex(value);
    return originalCallback?.apply(this, arguments);
  };
}
