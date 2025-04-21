const socket = new WebSocket("ws://localhost:8080");  // Ensure the correct port

socket.addEventListener("open", () => {
  console.log("✅ client start connect");

});
socket.addEventListener("message", (event) => {
  try {
      const data = JSON.parse(event.data);
      console.log(`📩 Response from server (Client ${data.clientId}):`, data.response);
   
      const response = data.response;
      if (response.progress !== undefined) {
      updateinterface(response);
      }
      if (response.status == 'error') {
        if (response.message == 'list index out of range') {
          var ProcessingeModal = document.getElementById("ProcessingeModal");
          ProcessingeModal.classList.remove("show");
          ProcessingeModal.classList.add("hide");
          popup('please select the correct layer');
          }
        }
      if (response.status == "Completed") {
        var ProcessingeModal = document.getElementById("ProcessingeModal");
        ProcessingeModal.classList.remove("show");
        ProcessingeModal.classList.add("hide");
        }
      if (response.progress == 100) {
      if(Object.keys(response.reqdatajson).length != 0)
           addnewlayerfromserver(response.reqdatajson);
           console.log("✅ client receive data",response.reqdatajson);
      }
      // Display response in UI
      // document.getElementById("output").innerText = `Server: ${data.response}`;
  } catch (error) {
      console.error("❌ Error parsing server response:", error);
  }
});

socket.addEventListener("close", () => {
  console.log("🔴 WebSocket connection closed");
});

socket.addEventListener("error", (error) => {
  console.error("⚠️ WebSocket error:", error);
});


function backgroundElem(elem) {
  let bg = document.getElementById('background');
  bg.appendChild(elem);
}
const dpr = window.devicePixelRatio || 1;
let input_clusters =2;
let input_src= "en";

let fps = 15;
let max_size = 4000 * 1e6 / 4; // 4GB max
let src='en';
let clusters=2;
// todo: add more types
const ext_map = {
  'mp4': 'video/mp4',
  'mpeg4': 'video/mp4',
  'mpeg': 'video/mpeg',
  'ogv': 'video/ogg',
  'webm': 'video/webm',
  'gif': 'image/gif',
  'jpg': 'image/jpeg',
  'jpeg': 'image/jpeg',
  'png': 'image/png',
  'webp': 'image/webp',
  'aac': 'audio/aac',
  'mp3': 'audio/mpeg',
  'oga': 'audio/ogg',
  'wav': 'audio/wav',
  'weba': 'audio/webm',
  };

  class Settings {
    constructor(onApply, popupDiv) {
      this.div = document.createElement('div');
      this.div.classList.toggle('settings');
  
      this.holder = document.createElement('div');
      this.holder.classList.toggle('holder');
      this.div.appendChild(this.holder);
  
      const ok = document.createElement('a');
      ok.textContent = '[apply]';
      ok.href = '#'; // Prevents unwanted navigation
      this.div.appendChild(ok);
  
      // Apply function to be executed when clicking apply
      ok.addEventListener('click', (event) => {
        event.preventDefault(); // Prevent default anchor behavior
        if (onApply) onApply(); // Execute the passed callback function
        if (popupDiv) popupDiv.div.remove(); // Close the popup
        this.close(); 
      });
  
      document.body.appendChild(this.div);
    }
  
    add(name, type, init, callback, elem_type = 'input') {
      let label = document.createElement('label');
      label.textContent = name;
  
      let setting = document.createElement(elem_type);
      setting.addEventListener('change', callback);
  
      if (type) {
        setting.type = type;
      }
  
      init(setting);
      this.holder.appendChild(label);
      this.holder.appendChild(setting);
    }
    close() {
      this.div.remove(); // Removes the settings panel from the DOM
    }
  }
  
  
function updateSettings() {
  const input_fps = document.getElementById("input_fps").value;
  const input_ram = document.getElementById("input_ram").value;
  max_size = 1e6 * Number.parseInt(input_ram.value);
  fps=Number.parseInt(input_fps.value);
  var settingModal = new bootstrap.Modal(document.getElementById("settingModal"));
  settingModal.hide();

}
function transcribe() {
  let settings = new Settings(() => {
    uploadAndProcessVideotranscribe(src, clusters); // Execute on Apply
  });

  settings.add('Source Language', 'text',
    e => e.value = src,
    e => src = e.target.value
  );

  settings.add('Expected number of speakers', 'text',
    e => e.value = clusters,
    e => clusters = Number.parseInt(e.target.value)
  );

  let popupDiv = popup(settings.div); // Store popup div reference
  settings.popupDiv = popupDiv; // Attach popup reference to settings
  
}

function exportToJson() {
  var xhr = new XMLHttpRequest();
  const date = new Date().getTime();
  const str = date + "_" + Math.floor(Math.random() * 1000) + ".json";
  // const url = "https://jott.live/save/note/"+ str + "/mebm";
  // xhr.open("POST", url, true);
  // xhr.setRequestHeader('Content-Type', 'application/json');
  // xhr.send(JSON.stringify({
  //   note: player.dumpToJson()
  // }));


  let uri = encodeURIComponent("https://jott.live/raw/" + str);
  let mebm_url = window.location + "#" + uri;
  const text = document.createElement('div');
  const preamble = document.createElement('span');
  preamble.textContent = "copy the link below to share:";
  const a = document.createElement('a');
  a.href = mebm_url;
  a.setAttribute("target", "_blank");
  a.textContent = "[link]";

  const json = document.createElement('pre');
  json.textContent = player.dumpToJson();
  json.style.overflow = 'scroll';
  json.style.wordBreak = 'break-all';
  json.style.height = '50%';
  text.appendChild(preamble);
  text.appendChild(document.createElement('br'));
  text.appendChild(document.createElement('br'));
  text.appendChild(a);
  text.appendChild(document.createElement('br'));
  text.appendChild(document.createElement('br'));
  const preamble2 = document.createElement('span');
  preamble2.textContent = "or save and host the JSON below";
  text.appendChild(preamble2);
  text.appendChild(document.createElement('br'));
  text.appendChild(document.createElement('br'));
  text.appendChild(json);
  popup(text);

 
}


class RenderedLayer {
  constructor(file) {
    this.name = file.name;
    if (file.uri) {
      this.uri = file.uri;
    }
    this.ready = false;

    this.total_time = 0;
    this.start_time = 0;

    this.width = 0;
    this.height = 0;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    backgroundElem(this.canvas);
  }

  dump() {
    return {
      width: this.width,
      height: this.height,
      name: this.name,
      start_time: this.start_time,
      total_time: this.total_time,
      uri: this.uri,
      type: this.constructor.name
    };
  }

  resize() {
    this.thumb_canvas.width = this.thumb_canvas.clientWidth * dpr;
    this.thumb_canvas.height = this.thumb_canvas.clientHeight * dpr;
    this.thumb_ctx.scale(dpr, dpr);
  }

  show_preview(ref_time) {
    if (!this.ready) {
      return;
    }
    this.thumb_ctx.clearRect(0, 0, this.thumb_canvas.clientWidth, this.thumb_canvas.clientHeight);
    this.render(this.thumb_ctx, ref_time);
  }


  update_name(name) {
    this.name = name;
    this.description.textContent = "\"" + this.name + "\"";
  }
  update_url(url) {
    this.uri = url;
   
  }
  init(player, preview) {
    this.player = player;
    this.preview = preview;
    this.canvas.width = this.player.width;
    this.canvas.height = this.player.height;
    this.title_div = this.preview.querySelector('.preview_title');

    this.description = document.createElement('span');
    this.description.classList.toggle('description');
    this.description.addEventListener('click', (function(e) {
      const new_text = prompt("enter new text");
      if (new_text) {
        this.update_name(new_text);
      }
    }).bind(this));
    this.title_div.appendChild(this.description);

    let delete_option = document.createElement('a');
    delete_option.textContent = '[x]';
    delete_option.style.float = "right";
    delete_option.addEventListener('click', (function() {
      if (confirm("delete layer \"" + this.name + "\"?")) {
        this.player.remove(this);
      }
    }).bind(this));
    this.title_div.appendChild(delete_option);

    this.thumb_canvas = this.preview.querySelector('.preview_thumb');
    this.thumb_ctx = this.thumb_canvas.getContext('2d');
    this.thumb_ctx.scale(dpr, dpr);
    this.update_name(this.name);
  }

  render_time(ctx, y_coord, width, selected) {
    let scale = ctx.canvas.clientWidth / this.player.total_time;
    let start = scale * this.start_time;
    let length = scale * this.total_time;
    if (selected) {
      ctx.fillStyle = `rgb(210,210,210)`;
    } else {
      ctx.fillStyle = `rgb(110,110,110)`;
    }
    ctx.fillRect(start, y_coord - width / 2, length, width);
    let end_width = width * 6;
    let tab_width = 2;
    ctx.fillRect(start, y_coord - end_width / 2, tab_width, end_width);
    ctx.fillRect(start + length - tab_width / 2, y_coord - end_width / 2, tab_width, end_width);
  }

  // default ignore drags, pinches
  update(change, time) {
    return;
  }

  drawScaled(ctx, ctx_out, video = false) {
    const width = video ? ctx.videoWidth : ctx.canvas.clientWidth;
    const height = video ? ctx.videoHeight : ctx.canvas.clientHeight;
    const in_ratio = width / height;
    const out_ratio = ctx_out.canvas.clientWidth / ctx_out.canvas.clientHeight;
    let ratio = 1;
    let offset_width = 0;
    let offset_height = 0;
    
    if (in_ratio > out_ratio) { // video is wider
      // match width
      ratio = ctx_out.canvas.clientWidth / width;
      offset_height = (ctx_out.canvas.clientHeight - (ratio * height)) / 2;
    } else { // out is wider
      // match height
      ratio = ctx_out.canvas.clientHeight / height;
      offset_width = (ctx_out.canvas.clientWidth - (ratio * width)) / 2;
    }
    ctx_out.drawImage((video ? ctx : ctx.canvas),
      0, 0, width, height,
      offset_width, offset_height, ratio * width, ratio * height);
  }
  getSelectedLayerFile() {
    console.log(`File/Path for selected layer: ${this.uri}`);
      return this.uri;
  }
  getSelectedLayername() {
    console.log(`File/Path for selected layer: ${this.name}`);
      return this.name;
  }
}

class MoveableLayer extends RenderedLayer {
  constructor(file) {
    super(file);
    // all moveables 2 seconds default
    this.total_time = 2 * 1000;
    this.frames = [];
    for (let i = 0; i < (this.total_time / 1000) * fps; ++i) {
      // x, y, scale, rot, anchor(bool)
      let f = new Float32Array(5);
      f[2] = 1;
      this.frames.push(f);
    }
    this.frames[0][4] = 1;
  }

  dump() {
    let obj = super.dump();
    obj.frames = [];
    for (let f of this.frames) {
      obj.frames.push(Array.from(f));
    }
    return obj;
  }

  adjustTotalTime(diff) {
    this.total_time += diff;
    const num_frames = Math.floor((this.total_time / 1000) * fps - this.frames.length);
    if (num_frames > 0) {
      for (let i = 0; i < num_frames; ++i) {
        let f = new Float32Array(5);
        f[2] = 1; // scale
        this.frames.push(f);
      }
    } else if (num_frames < 0) {
      // prevent overflow
      this.frames.splice(this.frames.length + num_frames + 1, 1 - num_frames);
    }
    const last_frame_time = this.getTime(this.frames.length - 1);
    const prev_anchor = this.nearest_anchor(last_frame_time, false);
    if (prev_anchor >= 0) {
      this.interpolate(prev_anchor);
    } else {
      this.interpolate(0);
    }
  }

  set_anchor(index) {
    this.frames[index][4] = 1;
  }

  is_anchor(index) {
    return this.frames[index][4];
  }

  delete_anchor(ref_time) {
    let i = this.getIndex(ref_time);
    this.frames[i][4] = 0;
    let prev_i = this.nearest_anchor(ref_time, false);
    this.interpolate(prev_i);
  }

  nearest_anchor(time, fwd) {
    if (this.getFrame(time)) {
      let i = this.getIndex(time);
      let inc = function() {
        if (fwd) {
          i++;
        } else {
          i--;
        }
      };
      inc();
      while (i >= 0 && i < this.frames.length) {
        if (this.is_anchor(i)) {
          return i;
        }
        inc();
      }
    }
    return -1;
  }

  // interpolates f1 into f0
  interpolate_frame(f0, f1, weight) {
    if (weight > 1) {
      weight = 1;
    } else if (weight < 0) {
      weight = 0;
    }
    let f = new Float32Array(5);
    f[0] = weight * f0[0] + (1 - weight) * f1[0];
    f[1] = weight * f0[1] + (1 - weight) * f1[1];
    f[2] = weight * f0[2] + (1 - weight) * f1[2];
    f[3] = weight * f0[3] + (1 - weight) * f1[3];
    return f;
  }

  // set index, k (of x, y, scale, rot) to val
  interpolate(index) {
    let frame = this.frames[index];
    let is_anchor = this.is_anchor(index);
    // find prev anchor
    let prev_idx = 0;
    let prev_frame = frame;
    let prev_is_anchor = false;
    let next_idx = this.frames.length - 1;
    let next_frame = frame;
    let next_is_anchor = false;

    for (let i = index - 1; i >= 0; i--) {
      if (this.is_anchor(i)) {
        prev_idx = i;
        prev_is_anchor = true;
        prev_frame = this.frames[i];
        break;
      }
    }

    for (let i = index + 1; i < this.frames.length; ++i) {
      if (this.is_anchor(i)) {
        next_idx = i;
        next_is_anchor = true;
        next_frame = this.frames[i];
        break;
      }
    }

    let prev_range = index - prev_idx;
    const eps = 1e-9;
    for (let i = 0; i <= prev_range; ++i) {
      let s = i / (prev_range + eps);
      this.frames[index - i] = this.interpolate_frame(prev_frame, frame, s);
    }
    let next_range = next_idx - index;
    for (let i = 0; i <= next_range; ++i) {
      let s = i / (next_range + eps);
      this.frames[index + i] = this.interpolate_frame(next_frame, frame, s);
    }
    if (prev_is_anchor) {
      this.set_anchor(prev_idx);
    }
    if (next_is_anchor) {
      this.set_anchor(next_idx);
    }
    if (is_anchor) {
      this.set_anchor(index);
    }
  }

  getIndex(ref_time) {
    let time = ref_time - this.start_time;
    let index = Math.floor(time / 1000 * fps);
    return index;
  }

  getTime(index) {
    return (index / fps * 1000) + this.start_time;
  }

  getFrame(ref_time) {
    let index = this.getIndex(ref_time);
    if (index < 0 || index >= this.frames.length) {
      return null;
    }
    let frame = new Float32Array(this.frames[index]);
    // we floored the index, but might need to interpolate subframes
    if (index + 1 < this.frames.length) {
      const diff = ref_time - this.getTime(index);
      const diff_next = this.getTime(index + 1) - ref_time;
      let next_frame = this.frames[index + 1];
      let s = diff_next / (diff + diff_next);
      frame = this.interpolate_frame(frame, next_frame, s);
    }
    return frame;
  }

  update(change, ref_time) {
    let f = this.getFrame(ref_time);
    if (!f) {
      return;
    }
    let index = this.getIndex(ref_time);
    if (change.scale) {
      const old_scale = f[2];
      const new_scale = f[2] * change.scale;
      let delta_x = ((this.width * old_scale) - (this.width * new_scale)) / 2;
      let delta_y = ((this.height * old_scale) - (this.height * new_scale)) / 2;
      this.frames[index][0] = f[0] + delta_x;
      this.frames[index][1] = f[1] + delta_y;
      this.frames[index][2] = new_scale;
      this.interpolate(index);
      this.set_anchor(index);
    }
    if (change.x) {
      this.frames[index][0] = change.x;
      this.interpolate(index);
      this.set_anchor(index);
    }
    if (change.y) {
      this.frames[index][1] = change.y;
      this.interpolate(index);
      this.set_anchor(index);
    }
    if (change.rotation) {
      this.frames[index][3] = f[3] + change.rotation;
      this.interpolate(index);
      this.set_anchor(index);
    }
  }

  // moveable layers have anchor points we'll want to show
  render_time(ctx, y_coord, base_width, selected) {
    super.render_time(ctx, y_coord, base_width, selected);
    let scale = ctx.canvas.clientWidth / this.player.total_time;
    let width = 4 * base_width;
    for (let i = 0; i < this.frames.length; ++i) {
      if (this.is_anchor(i)) {
        let anchor_x = this.start_time + 1000 * (i / fps);
        ctx.fillStyle = `rgb(100,210,255)`;
        ctx.fillRect(scale * anchor_x, y_coord - width / 2, 3, width);
      }
    }
  }

}

class ImageLayer extends MoveableLayer {
  constructor(file) {
    super(file);
    // assume images are 10 seconds
    this.img = new Image();

    this.reader = new FileReader();
    this.reader.addEventListener("load", (function() {
      this.img.src = this.reader.result;
      this.img.addEventListener('load', (function() {
        this.width = this.img.naturalWidth;
        this.height = this.img.naturalHeight;
        this.ready = true;
      }).bind(this));
    }).bind(this), false);
    this.reader.readAsDataURL(file);
  }

  render(ctx_out, ref_time) {
    if (!this.ready) {
      return;
    }
    let f = this.getFrame(ref_time);
    if (f) {
      let scale = f[2];
      let x = f[0] + this.canvas.width / 2 - this.width / 2;
      let y = f[1] + this.canvas.height / 2 - this.height / 2;
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.drawImage(this.img, 0, 0, this.width, this.height, x, y, scale * this.width, scale * this.height);
      this.drawScaled(this.ctx, ctx_out);
    }
  }
}

class TextLayer extends MoveableLayer {
  constructor(text) {
    let f = {
      name: text
    };
    super(f);
    this.color = "#ffffff";
    this.shadow = true;
    this.ready = true;
  }

  init(player, preview) {
    super.init(player, preview);

    let settings = new Settings();

    settings.add('text', null,
      i => i.value = this.name,
      e => this.update_name(e.target.value),
      'textarea'
    );

    settings.add('color', 'color',
      i => i.value = this.color,
      e => this.color = e.target.value
    );

    settings.add('shadow', 'checkbox',
      i => i.checked = this.shadow,
      e => this.shadow = e.target.checked
    );

    let settings_link = document.createElement('a');
    settings_link.style.float = "right";
    settings_link.textContent = "[...]";
    settings_link.addEventListener('click', function() {
      popup(settings.div);
    });
    this.title_div.appendChild(settings_link);

  }

  update(change, ref_time) {
    let rect = this.ctx.measureText(this.name);
    this.width = rect.width;
    this.height = rect.actualBoundingBoxAscent + rect.actualBoundingBoxDescent;
    super.update(change, ref_time);
  }

  render(ctx_out, ref_time) {
    let f = this.getFrame(ref_time);
    if (f) {

      let scale = f[2];
      this.ctx.font = Math.floor(scale * 30) + "px Georgia";
      let lines = this.name.split('\n');
      let rect = this.ctx.measureText(this.name);
      this.width = rect.width;
      this.height = rect.actualBoundingBoxAscent + rect.actualBoundingBoxDescent;
      let x = f[0] + this.canvas.width / 2;
      let y = f[1] + this.canvas.height / 2;
      if (this.shadow) {
        this.ctx.shadowColor = "black";
        this.ctx.shadowBlur = 7;
      } else {
        this.ctx.shadowColor = null;
        this.ctx.shadowBlur = null;
      }
      this.ctx.fillStyle = this.color;
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.save();
      this.ctx.translate(x, y);
      this.ctx.rotate(f[3] * (Math.PI / 180));
      this.ctx.textAlign = "center";
      this.ctx.fillText(this.name, 0, 0);
      this.ctx.restore();
      this.drawScaled(this.ctx, ctx_out);
    }
  }
}

class VideoLayer extends RenderedLayer {
  constructor(file) {
    super(file);
    this.frames = [];
    // for creating empty VideoLayers (split() requires this)
    if (file._leave_empty) {
      return;
    }

    // assume all videos fit in 4GB of ram
    this.video = document.createElement('video');
    this.video.setAttribute('autoplay', true);
    this.video.setAttribute('loop', true);
    this.video.setAttribute('playsinline', true);
    this.video.setAttribute('muted', true);
    this.video.setAttribute('controls', true);
    backgroundElem(this.video);

    this.reader = new FileReader();
    this.reader.addEventListener("load", (function() {
      this.video.addEventListener('loadedmetadata', (function() {
        let width = this.video.videoWidth;
        let height = this.video.videoHeight;
        let dur = this.video.duration;
        this.total_time = dur * 1000;
        let size = fps * dur * width * height;
        if (size < max_size) {
          this.width = width;
          this.height = height;
        } else {
          let scale = size / max_size;
          this.width = Math.floor(width / scale);
          this.height = Math.floor(height / scale);
        }
        const player_ratio = this.player.width / this.player.height;
        const video_ratio = this.width / this.height;
        if (video_ratio > player_ratio) { // video is wider, make it taller
          let scale = video_ratio / player_ratio;
          this.height *= scale;
        } else {
          let scale = player_ratio / video_ratio;
          this.width *= scale;
        }
        this.canvas.height = this.height;
        this.canvas.width = this.width;
        this.convertToArrayBuffer();
      }).bind(this));
      this.video.src = this.reader.result;
    }).bind(this), false);
    this.reader.readAsDataURL(file);
  }

  async seek(t) {
    return await (new Promise((function(resolve, reject) {
      this.video.currentTime = t;
      this.video.pause();
      this.video.addEventListener('seeked', (function(ev) {
        this.drawScaled(this.video, this.ctx, true);
        this.thumb_canvas.width = this.thumb_canvas.clientWidth * dpr;
        this.thumb_canvas.height = this.thumb_canvas.clientHeight * dpr;
        this.thumb_ctx.clearRect(0, 0, this.thumb_canvas.clientWidth, this.thumb_canvas.clientHeight);
        this.thumb_ctx.scale(dpr, dpr);
        this.drawScaled(this.ctx, this.thumb_ctx);
        let frame = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
        resolve(frame);
      }).bind(this), {
        once: true
      });
    }).bind(this)));
  }

  async convertToArrayBuffer() {
    this.video.pause();
    let d = this.video.duration;
    let name = this.name;
    for (let i = 0; i < d * fps; ++i) {
      let frame = await this.seek(i / fps);
      let sum = 0;
      for (let j = 0; j < frame.data.length; ++j) {
        sum += frame.data[j];
      }
      this.frames.push(frame);
      this.update_name((100 * i / (d * fps)).toFixed(2) + "%");
    }
    this.ready = true;
    this.video.remove();
    this.video = null;
    this.update_name(name);
  }

  render(ctx_out, ref_time) {
    if (!this.ready) {
      return;
    }
    let time = ref_time - this.start_time;
    let index = Math.floor(time / 1000 * fps);
    if (index < this.frames.length) {
      const frame = this.frames[index];
      this.ctx.putImageData(frame, 0, 0);
      this.drawScaled(this.ctx, ctx_out);
    }
  }
}

var AudioContext = window.AudioContext // Default
  ||
  window.webkitAudioContext // Safari and old versions of Chrome
  ||
  false;
class AudioLayer extends RenderedLayer {
  constructor(file) {
    super(file);
    this.reader = new FileReader();
    this.audio_ctx = new AudioContext();
    this.audio_buffer = null;
    this.source = null;
    this.playing = false;
    this.last_time = 0;
    this.last_ref_time = 0;
    this.reader.addEventListener("load", (function() {
      let buffer = this.reader.result;
      this.audio_ctx.decodeAudioData(buffer, (aud_buffer) => {
        this.audio_buffer = aud_buffer;
        this.total_time = this.audio_buffer.duration * 1000;
        if (this.total_time === 0) {
          this.player.remove(this);
        }
        this.ready = true;
      }, (function(e) {
        this.player.remove(this);
      }).bind(this));
    }).bind(this));
    this.reader.readAsArrayBuffer(file);
  }

  disconnect() {
    if (this.source) {
      this.source.disconnect(this.player.audio_ctx.destination);
    }
  }

  init_audio(ref_time) {
    this.disconnect();
    this.source = this.player.audio_ctx.createBufferSource();
    this.source.buffer = this.audio_buffer;
    this.source.connect(this.player.audio_ctx.destination);
    if (this.player.audio_dest) {
      this.source.connect(this.player.audio_dest);
    }
    this.started = false;
  }

  init(player, preview) {
    super.init(player, preview);
  }

  update_name(name) {
    this.name = name;
    this.description.textContent = "\"" + this.name + "\" [audio]";
  }

  render(ctx_out, ref_time) {
    if (!this.ready) {
      return;
    }
    if (!this.player.playing) {
      return;
    }
    let time = ref_time - this.start_time;
    if (time < 0 || time > this.total_time) {
      return;
    }
    if (!this.started) {
      if (!this.source) {
        this.init_audio(ref_time);
      }
      this.source.start(0, time / 1000);
      this.started = true;
    }
  }
};

class Player {

  constructor() {
    this.playing = false;
    this.scrubbing = false;
    this.layers = [];
    this.selected_layer = null;
    this.onend_callback = null;
    this.update = null;
    this.width = 1280;
    this.height = 720;
    this.total_time = 0;
    this.last_step = null;
    this.time = 0;
    this.last_paused = Number.MAX_SAFE_INTEGER;
    // for preview
    this.aux_time = 0;
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.audio_ctx = new AudioContext();
    this.canvas_holder = document.getElementById('canvas');
    this.canvas_holder.appendChild(this.canvas);
    this.time_scale = 1.0;
    this.time_holder = document.getElementById('time');
    this.time_canvas = document.createElement('canvas');
    this.time_canvas.addEventListener('pointerdown', this.scrubStart.bind(this));
    this.time_canvas.addEventListener('pointermove', this.scrubMove.bind(this), {passive:false});
    this.time_canvas.addEventListener('pointerleave', this.scrubEnd.bind(this));
    this.time_ctx = this.time_canvas.getContext('2d');
    this.time_holder.appendChild(this.time_canvas);
    this.cursor_preview = document.getElementById('cursor_preview');
    this.cursor_canvas = this.cursor_preview.querySelector('canvas');
    this.cursor_ctx = this.cursor_canvas.getContext('2d');
    this.cursor_text = this.cursor_preview.querySelector('div');
    window.requestAnimationFrame(this.loop.bind(this));
   
    this.setupPinchHadler(this.canvas_holder,
      (function(scale, rotation) {
        this.update = {
          scale: scale,
          rotation: rotation
        };
      }).bind(this));
    this.setupPinchHadler(this.time_holder,
      (function(scale, rotation) {
       let new_x = (this.time_holder.clientWidth * scale - this.time_holder.clientWidth);
       let old_x = this.time_holder.scrollLeft;
       this.time_scale = Math.max(1, this.time_scale * scale);
       this.resize_time();
       this.time_holder.scroll(Math.round(old_x + new_x), 0);
      }).bind(this));
    this.setupDragHandler();
    this.resize();
  }

  dumpToJson() {
    let out = [];
    for (let layer of this.layers) {
      out.push(layer.dump());
    }
    return JSON.stringify(out);
  }
  playForward() {
    player.next();
  }

  playBackward() {
    player.prev();
  }

  async loadFromJson(data) {
    try {
      // let data = JSON.parse(jsonString);
      
      if (!Array.isArray(data)) {
        console.error("Invalid JSON format: Expected an array of layers.");
        return;
      }
  
      // Reset current state
      this.layers = [];
      this.time = 0;
      this.total_time = 0;
  
      // Load layers
      await this.loadLayersfromaerver(data);
    //   for (let layer_d of data) {
    //   //   if (layer_d.type === "VideoLayer") {
    //   // // Restore other properties if they exist
    //   // if (data.width) this.width = layer_d.width;
    //   // if (data.height) this.height = layer_d.height;
    //   // if (data.total_time) this.total_time = layer_d.total_time;
    //   // if (data.time) this.time = layer_d.time;
    //   // }
    // }
      console.log("Player successfully loaded from JSON.");
    } catch (error) {
      console.error("Error loading from JSON:", error);
    }
  }
  async loadLayersfromaerver(layers) {
    // Function to wait until a layer is ready
    let on_ready = function(d, c) {
        if (!d.ready) {
            setTimeout(function(){ on_ready(d, c); }, 10);
        } else {
            c(d);
        }
    };

    // Iterate through each layer
    for (let layer_d of layers) {
        let layer = null;

        // Handle VideoLayer
        if (layer_d.type == "VideoLayer") {
            layer = await this.addURIfromserver(layer_d.uri);  // Fetch video from the server
        }
        // Handle TextLayer
        else if (layer_d.type == "TextLayer") {
            layer = this.add(new TextLayer(layer_d.name));
        }
        // Handle ImageLayer
        else if (layer_d.type == "ImageLayer") {
            layer = await this.addURIfromserver(layer_d.uri);  // Fetch image from the server
        }
        // Handle other types like AudioLayer (currently commented out)
        // else if (layer_d.type == "AudioLayer") {
        //     layer = await this.addURI(layer_d.uri);
        // }

        // If the layer is not created, log an error and continue to the next layer
        if (!layer) {
            console.log("layer couldn't be processed");
            continue;
        }

        // Ensure the layer is ready and set its properties
        on_ready(layer, function(l) {
            l.name = layer_d.name;
            l.width = layer_d.width;
            l.height = layer_d.height;
            l.start_time = layer_d.start_time;
            l.total_time = layer_d.total_time;

            // If there are frames data, convert it to Float32Array
            if (layer_d.frames) {
                l.frames = [];
                for (let f of layer_d.frames) {
                    l.frames.push(new Float32Array(f));
                }
            }
        });
    }
}

  async loadLayers(layers) {
    let on_ready = function(d, c) {
      if (!d.ready) {
        setTimeout(function(){ on_ready(d, c); }, 10);
      } else {
        c(d);
      }
    };
    for (let layer_d of layers) {
      let layer = null;
      if (layer_d.type == "VideoLayer") {
        layer = await this.addURI(layer_d.uri);
      } else if (layer_d.type == "TextLayer") {
        layer = this.add(new TextLayer(layer_d.name));
      } else if (layer_d.type == "ImageLayer") {
        layer = await this.addURI(layer_d.uri);
      }
      //  this.add(new AudioLayer(file));
      if (!layer) {
        console.log("layer couldn't be processed");
        continue;
      }
      on_ready(layer, function(l) {
        layer.name = layer.name;
        layer.width = layer_d.width,
        layer.height = layer_d.height,
        layer.start_time = layer_d.start_time;
        layer.total_time = layer_d.total_time;
        if (layer_d.frames) {
          layer.frames = [];
          for (let f of layer_d.frames) {
            layer.frames.push(new Float32Array(f));
          }
        }
      });
    }
  }

  intersectsTime(time, query) {
    if (!query) {
      query = this.time;
    }
    return Math.abs(query - time) / this.total_time < 0.01;
  }

  refresh_audio() {
    for (let layer of this.layers) {
      if (layer instanceof AudioLayer) {
        layer.init_audio(this.time);
      }
    }
  }

  play() {
    this.playing = true;
    if (this.last_paused != this.time) {
      this.refresh_audio();
    }
    this.audio_ctx.resume();
  }

  pause() {
    this.playing = false;
    this.audio_ctx.suspend();
    this.last_paused = this.time;
  }

  scrubStart(ev) {
    this.scrubbing = true;
    let rect = this.time_canvas.getBoundingClientRect();
    this.time = ev.offsetX / rect.width * this.total_time;

    window.addEventListener('pointerup', this.scrubEnd.bind(this), {
      once: true
    });

    let y_inc = this.time_canvas.clientHeight / (this.layers.length + 1);
    let y_coord = this.time_canvas.clientHeight;
    let mouseover = false;
    for (let layer of this.layers) {
      y_coord -= y_inc;
      if (layer.start_time > (1.01 * this.time)) {
        continue;
      }
      if (layer.start_time + layer.total_time < (0.99 * this.time)) {
        continue;
      }
      if (Math.abs(ev.offsetY - y_coord) < (0.05 * this.time_canvas.clientHeight)) {
        this.select(layer);
        mouseover = true;
      }
    }

    // can't drag unselected
    if (!this.selected_layer || !mouseover) {
      return;
    }

    // dragging something
    let l = this.selected_layer;

    if (this.intersectsTime(l.start_time)) {
      this.time = l.start_time;
      let base_t = this.time;
      this.dragging = function(t) {
        let diff = t - base_t;
        base_t = t;
        l.start_time += diff;
      }
    } else if (this.intersectsTime(l.start_time + l.total_time)) {
      this.time = l.start_time + l.total_time;
      let base_t = this.time;
      this.dragging = function(t) {
        let diff = t - base_t;
        base_t = t;
        if (l instanceof MoveableLayer) {
          l.adjustTotalTime(diff);
        } else {
          l.start_time += diff;
        }
      }
    } else if (this.time < l.start_time + l.total_time && this.time > l.start_time) {
      let base_t = this.time;
      this.dragging = function(t) {
        let diff = t - base_t;
        base_t = t;
        l.start_time += diff;
      }
    }
  }
  updateCursorText() {
    let milliseconds = Math.floor(this.aux_time / 1000); 
    let millitotalSeconds = Math.floor(this.total_time / 1000); 
    let minutes = Math.floor(milliseconds/ 60);
    let seconds = Math.floor(milliseconds % 60);
    let totalMinutes = Math.floor(millitotalSeconds / 60);
    let totalSeconds = Math.floor(millitotalSeconds % 60);

    // Format as MM:SS
    let formattedTime = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    let formattedTotalTime = `${totalMinutes}:${totalSeconds.toString().padStart(2, '0')}`;

    this.cursor_text.textContent = `${formattedTime}/${formattedTotalTime}`;
}

  scrubMove(ev) {
    ev.preventDefault();
    ev.stopPropagation();
    let rect = this.time_canvas.getBoundingClientRect();
    let time = ev.offsetX / rect.width * this.total_time;

    document.body.style.cursor = "default";

    if (this.selected_layer) {
      let l = this.selected_layer;
      if (this.intersectsTime(l.start_time, time)) {
        document.body.style.cursor = "col-resize";
      }
      if (this.intersectsTime(l.start_time + l.total_time, time)) {
        document.body.style.cursor = "col-resize";
      }
    }
    this.cursor_preview.style.position = "absolute";
    this.cursor_preview.style.zIndex  = 1000000;

    this.cursor_preview.style.display = "block";
    let cursor_x = Math.max(ev.clientX - this.cursor_canvas.clientWidth / 2, 0);
    cursor_x = Math.min(cursor_x, rect.width - this.cursor_canvas.clientWidth);
    this.cursor_preview.style.left = cursor_x + "px";
    this.cursor_preview.style.bottom = (rect.height) + "px";
    this.cursor_preview.style.color = "#8d8d8d";

    this.aux_time = time;
    this.updateCursorText();
    // this.cursor_text.textContent = this.aux_time.toFixed(2) + "/" + this.total_time.toFixed(2)


    if (this.scrubbing) {
      this.time = time;
    }

    if (this.dragging) {
      this.dragging(this.time);
    }
  }

  scrubEnd(ev) {
    document.body.style.cursor = "default";
    this.cursor_preview.style.display = "none";
    this.scrubbing = false;
    this.dragging = null;
    this.total_time = 0;
    this.aux_time = 0;
  }

  setupPinchHadler(elem, callback) {
    // safari only
    let gestureStartRotation = 0;
    let gestureStartScale = 0;

    let wheel = function(e) {
      if (e.ctrlKey || e.shiftKey) {
        e.preventDefault();
        let delta = e.deltaY;
        if (!Math.abs(delta) && e.deltaX != 0) {
          delta = e.deltaX * 0.5;
        }
        let scale = 1;
        scale -= delta * 0.01;
        // Your zoom/scale factor
        callback(scale, 0);
      } else if (e.altKey) {
        let delta = e.deltaY;
        if (!Math.abs(delta) && e.deltaX != 0) {
          delta = e.deltaX * 0.5;
        }
        let rot = -delta * 0.1;
        // Your zoom/scale factor
        callback(0, rot);
      }
    }
    // safari
    let gesturestart = function(e) {
      this.gesturing = true;
      e.preventDefault();
      gestureStartRotation = e.rotation;
      gestureStartScale = e.scale;
    };
    let gesturechange = function(e) {
      e.preventDefault();
      e.stopPropagation();
      let rotation = e.rotation - gestureStartRotation;
      let scale = e.scale / gestureStartScale;
      gestureStartRotation = e.rotation;
      gestureStartScale = e.scale;
      callback(scale, rotation);
    };
    let gestureend = function(e) {
      this.gesturing = false;
      e.preventDefault();
    };
    elem.addEventListener('gesturestart', gesturestart.bind(this));
    elem.addEventListener('gesturechange', gesturechange.bind(this));
    elem.addEventListener('gestureend', gestureend.bind(this));
    // everyone else
    elem.addEventListener('wheel', wheel.bind(this), {
      passive: false
    });
    let deleter = function() {
      elem.removeEventListener('gesturestart', gesturestart);
      elem.removeEventListener('gesturechange', gesturechange);
      elem.removeEventListener('gestureend', gestureend);
      elem.removeEventListener('wheel', wheel);
    }
  }

  setupDragHandler() {
    let callback = (function(x, y) {
      this.update = {
        x: x,
        y: y
      };
    }).bind(this);
    let elem = this.canvas_holder;
    let dragging = false;
    let base_x = 0;
    let base_y = 0;
    let pointerup = function(e) {
      dragging = false;
      e.preventDefault();
    }
    let get_ratio = (function(elem) {
      let c_ratio = elem.clientWidth / elem.clientHeight;
      let target_ratio = this.width / this.height;
      // how many player pixels per client pixels
      let ratio = 1;
      if (c_ratio > target_ratio) { // client is wider than player
        ratio = this.height / elem.clientHeight;
      } else {
        ratio = this.width / elem.clientWidth;
      }
      return ratio;
    }).bind(this);
    let pointerdown = function(e) {
      if (!this.selected_layer) {
        return;
      }
      if (!(this.selected_layer instanceof MoveableLayer)) {
        return;
      }
      e.preventDefault();
      let f = this.selected_layer.getFrame(this.time);
      if (!f) {
        return;
      }
      dragging = true;
      base_x = e.offsetX * get_ratio(e.target) - f[0];
      base_y = e.offsetY * get_ratio(e.target) - f[1];
      window.addEventListener('pointerup', pointerup, {
        once: true
      });
    }
    let pointermove = function(e) {
      if (this.gesturing) { return; }
      e.preventDefault(); 
      e.stopPropagation();
      if (dragging) {
        let dx = e.offsetX * get_ratio(e.target) - base_x;
        let dy = e.offsetY * get_ratio(e.target) - base_y;
        callback(dx, dy);
      }
    }
    elem.addEventListener('pointerdown', pointerdown.bind(this));
    elem.addEventListener('pointermove', pointermove.bind(this), {passive:false});
    let deleter = function() {
      elem.removeEventListener('pointerdown', pointerdown);
      elem.removeEventListener('pointermove', pointermove);
    }
  }

  prev() {
    if (this.selected_layer) {
      let l = this.selected_layer;
      if (l instanceof MoveableLayer) {
        let i = l.nearest_anchor(this.time, false);
        if (i >= 0) {
          this.time = l.getTime(i);
          return;
        }
      }
    }
    this.time = Math.max(this.time - 100, 0);
  }

  next() {
    if (this.selected_layer) {
      let l = this.selected_layer;
      if (l instanceof MoveableLayer) {
        let i = l.nearest_anchor(this.time, true);
        if (i >= 0) {
          this.time = l.getTime(i);
          return;
        }
      }
    }
    this.time = Math.min(this.time + 100, this.total_time - 1);
  }

  delete_anchor() {
    if (this.selected_layer) {
      let l = this.selected_layer;
      if (l instanceof MoveableLayer) {
        l.delete_anchor(this.time);
        this.prev();
      }
    }
  }

  deselect() {
    if (this.selected_layer !== null) {
      this.selected_layer.preview.classList.toggle('selected');
    }
  }

  select(layer) {
    this.deselect();
    this.selected_layer = layer;
    this.selected_layer.preview.classList.toggle('selected');
  }
  removeall() {
    const len = this.layers.length;
  
    // Disconnect all AudioLayers and clear the layers array
    for (let layer of this.layers) {
      if (layer instanceof AudioLayer) {
        layer.disconnect();
      }
    }
    this.layers = [];
  
    // Remove all corresponding UI elements
    const layer_picker = document.getElementById('layers');
    while (layer_picker.firstChild) {
      layer_picker.removeChild(layer_picker.firstChild);
    }
  
    // Reset time tracking
    this.total_time = 0;
    this.time = 0;
  }
  
  remove(layer) {
    const idx = this.layers.indexOf(layer);
    const len = this.layers.length;
    if (idx > -1) {
      this.layers.splice(idx, 1);
      let layer_picker = document.getElementById('layers');
      // divs are reversed
      layer_picker.children[len - idx - 1].remove();
    }
    if (layer instanceof AudioLayer) {
      layer.disconnect();
    }
    this.total_time = 0;
    for (let layer of this.layers) {
      if (layer.start_time + layer.total_time > this.total_time) {
        this.total_time = layer.start_time + layer.total_time;
      }
    }
    if (this.time > this.total_time) {
      this.time = this.total_time;
    }
  }

  add(layer) {
    let layer_picker = document.getElementById('layers');

    // Create elements off-DOM
    let preview = document.createElement('div');
    let thumb = document.createElement('canvas');
    let title = document.createElement('div');

    // Use classList.add instead of toggle (faster for initial add)
    preview.classList.add('preview');
    thumb.classList.add('preview_thumb');
    title.classList.add('preview_title');

    preview.setAttribute('draggable', true);

    // Use arrow functions to avoid unnecessary `.bind(this)`
    preview.addEventListener('dragstart', (ev) => {
        this.preview_dragging = preview;
        this.preview_dragging_layer = layer;
    });

    preview.addEventListener('dragover', (ev) => ev.preventDefault());

    preview.addEventListener('drop', (ev) => {
        preview.before(this.preview_dragging);
        let idx = this.layers.indexOf(this.preview_dragging_layer);
        if (idx > -1) this.layers.splice(idx, 1);
        let new_idx = this.layers.indexOf(layer);
        this.layers.splice(new_idx + 1, 0, this.preview_dragging_layer);
        this.select(this.preview_dragging_layer);
        this.preview_dragging = null;
        this.preview_dragging_layer = null;
    });

    preview.addEventListener('click', () => this.select(layer));

    // Append elements together before inserting into the DOM
    preview.append(thumb, title);

    // Batch DOM updates by using document fragment
    let fragment = document.createDocumentFragment();
    fragment.appendChild(preview);
    layer_picker.prepend(fragment);

    // Layer initialization
    layer.start_time = this.time;
    layer.init(this, preview);

    // Push to array after modifications
    this.layers.push(layer);
    this.select(layer);

    return layer;
}

  split() {
    if (!this.selected_layer) {
      return;
    }
    let l = this.selected_layer;
    if (!(l instanceof VideoLayer)) {
      return;
    }
    if (!l.ready) {
      return;
    }
    if (l.start_time > this.time) {
      return;
    }
    if (l.start_time + l.total_time < this.time) {
      return;
    }
    let nl = new VideoLayer({
      name: l.name + "NEW",
      _leave_empty: true
    });
    const pct = (this.time - l.start_time) / l.total_time;
    const split_idx = Math.round(pct * l.frames.length);
    nl.frames = l.frames.splice(0, split_idx);
    this.add(nl);
    nl.start_time = l.start_time;
    nl.total_time = pct * l.total_time;
    l.start_time = l.start_time + nl.total_time;
    l.total_time = l.total_time - nl.total_time;
    nl.width = l.width;
    nl.height = l.height;
    nl.canvas.width = l.canvas.width;
    nl.canvas.height = l.canvas.height;
    nl.resize(); // fixup thumbnail
    nl.ready = true;
  }

  onend(callback) {
    this.onend_callback = callback;
  }

  render(ctx, time, update_preview) {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    for (let layer of this.layers) {
      if (layer.start_time > time) {
        continue;
      }
      if (layer.start_time + layer.total_time < time) {
        continue;
      }
      layer.render(ctx, time);
      if (update_preview) {
        layer.show_preview(time);
      }
    }
  }

  resize_time() {
    this.time_canvas.style.width = this.time_holder.clientWidth * this.time_scale;
    this.time_canvas.width = this.time_canvas.clientWidth * dpr;
    this.time_canvas.height = this.time_canvas.clientHeight * dpr;
    this.time_ctx.scale(dpr, dpr);
  }

  resize() {
    // update canvas and time sizes
    this.canvas.width = this.canvas.clientWidth * dpr;
    this.canvas.height = this.canvas.clientHeight * dpr;
    this.ctx.scale(dpr, dpr);
    this.resize_time();
    for (let layer of this.layers) {
      layer.resize();
    }
  }

  loop(realtime) {
    for (let layer of this.layers) {
      if (layer.start_time + layer.total_time > this.total_time) {
        this.total_time = layer.start_time + layer.total_time;
      }
    }
    // draw time
    if (this.last_step === null) {
      this.last_step = realtime;
    }
    if (this.playing && this.total_time > 0) {
      this.time += (realtime - this.last_step);
      if (this.onend_callback && this.time >= this.total_time) {
        this.onend_callback(this);
        this.onend_callback = null;
      }
      if (this.time >= this.total_time) {
        this.refresh_audio();
      }
      this.time %= this.total_time;
    }
    this.last_step = realtime;
    this.time_ctx.clearRect(0, 0, this.time_canvas.clientWidth, this.time_canvas.clientWidth);
    let x = this.time_canvas.clientWidth * this.time / this.total_time;
    this.time_ctx.fillStyle = `rgb(210,210,210)`;
    this.time_ctx.fillRect(x, 0, 2, this.time_canvas.clientHeight);
    this.time_ctx.font = "10px courier";

    let milliseconds = Math.floor(this.time / 1000); 
    let millitotalSeconds = Math.floor(this.total_time / 1000); 
    let minutes = Math.floor(milliseconds/ 60);
    let seconds = Math.floor(milliseconds % 60);
    let totalMinutes = Math.floor(millitotalSeconds / 60);
    let totalSeconds = Math.floor(millitotalSeconds % 60);

    // Format as MM:SS
    let formattedTime = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    let formattedTotalTime = `${totalMinutes}:${totalSeconds.toString().padStart(2, '0')}`;

    this.time_ctx.fillText(formattedTime+"", x + 5, 10);
    this.time_ctx.fillText(formattedTotalTime+"", x + 5, 20);

    if (this.aux_time > 0) {
      let aux_x = this.time_canvas.clientWidth * this.aux_time / this.total_time;
      this.time_ctx.fillStyle = `rgb(110,110,110)`;
      this.time_ctx.fillRect(aux_x, 0, 1, this.time_canvas.clientHeight);
      this.render(this.cursor_ctx, this.aux_time, false);
    }

    let y_inc = this.time_canvas.clientHeight / (this.layers.length + 1);
    let y_coord = this.time_canvas.clientHeight - y_inc;
    for (let layer of this.layers) {
      let selected = this.selected_layer == layer;
      layer.render_time(this.time_ctx, y_coord, 3, selected);
      y_coord -= y_inc;
      if (this.selected_layer == layer && this.update) {
        layer.update(this.update, this.time);
        this.update = null;
      }
    }
    this.render(this.ctx, this.time, true);
    window.requestAnimationFrame(this.loop.bind(this));
  }

  addFile(file,noaudio=false) {
    if (file.type.indexOf('video') >= 0 && noaudio==true ) {
      const layer= new VideoLayer(file);
      this.add(layer);
      return layer
    } else if (file.type.indexOf('video') >= 0) {
      this.add(new AudioLayer(file));
      const layer= new VideoLayer(file);
      this.add(layer);
      return layer
    } else if (file.type.indexOf('image') >= 0) {
      const layer= new ImageLayer(file);
      this.add(layer);
      return layer
    } else if (file.type.indexOf('audio') >= 0) {
      const layer= new AudioLayer(file);
      this.add(layer);
      return layer

    }
  }
  async addURIfromserver(uri) {
     // Safari bug workaround for URI issues
     if (!uri) {
      return;
  }

  // Extract file extension to determine the type of resource
  let extension = uri.split(/[#?]/)[0].split('.').pop().trim();

  // If the extension is not supported, handle 'json' extension or return
  if (!ext_map[extension]) {
      if (extension == 'json') {
          // Fetch JSON and load layers asynchronously
          let response = await fetch("/backend/uploads/"+uri);
          let layers = await response.json();
          player.loadLayers(layers); // Assuming player.loadLayers is globally accessible
      }
      return;
  }

  // Metadata object with the type based on extension mapping
  let metadata = {
      type: ext_map[extension]
  };

  // Extract the file name from the URI
  let segs = uri.split("/");
  let name = segs[segs.length - 1];
  // alert("/backend/uploads/"+uri);
  // Fetch the file from the URI
  let response = await fetch("http://localhost:8080/backend/uploads/" + uri);
  let data = await response.blob();

  // Create a File object with metadata
  let file = new File([data], name, metadata);
  file.uri = uri;

  // Add the file (this is where the layer creation happens)
  return this.addFile(file);
}

  async addURI(uri) {
    // safari has a bug here
    if (!uri) {
      return;
    }
    let extension = uri.split(/[#?]/)[0].split('.').pop().trim();

    if (!ext_map[extension]) {
      if (extension == 'json') {
        let response = await fetch(uri);
        let layers = await response.json();
        player.loadLayers(layers);
      }
      return;
    }
    let metadata = {
      type: ext_map[extension]
    };
    let segs = uri.split("/");
    let name = segs[segs.length - 1];
    let response = await fetch(uri);
    let data = await response.blob();
    let file = new File([data], name, metadata);
    file.uri = uri;
    return this.addFile(file);
  }

}


let player = new Player();

window.addEventListener("drop", async function (ev) {
  ev.preventDefault();

  if (ev.dataTransfer.items) {
      for (var i = 0; i < ev.dataTransfer.items.length; i++) {
          let item = ev.dataTransfer.items[i];

          if (item.kind === "file") {
              var file = item.getAsFile();
              console.log("Uploading file:", file.name);

              // Upload the file & get the server URL
              const fileUrl = await uploadFile(file);
              // file.name=fileUrl;
            
            

              // Use the file URL (e.g., display it in the player)
              var layer= player.addFile(file);
              layer.update_url(fileUrl);
              console.log("File uploaded to:", layer.uri);
          }
      }
  }
});

// 📌 Function to Upload the File and Get the URL
async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  try {
      const response = await fetch("http://localhost:8080/uploaddata", {
          method: "POST",
          body: formData,
      });

      const result = await response.json();
      return result.fileUrl; // 🔥 Returns the file URL from the server
  } catch (error) {
      console.error("Upload failed:", error);
      return null;
  }
}

window.addEventListener('paste', function(ev) {
  let uri = (event.clipboardData || window.clipboardData).getData('text');
  player.addURI(uri);
});

// TODO show something
window.addEventListener('dragover', function(e) {
  e.preventDefault();
});


window.addEventListener('keydown', function(ev) {
  if (ev.code == "Space") {
    if (player.playing) {
      player.pause();
    } else {
      player.play();
    }
  } else if (ev.code == "ArrowLeft") {
    player.prev();
  } else if (ev.code == "ArrowRight") {
    player.next();
  } else if (ev.code == "Backspace") {
    player.delete_anchor();
  } else if (ev.code == "KeyS") {
    player.split();
  } else if (ev.code == "KeyI") {
    if (ev.ctrlKey) {
      let uris = prompt("paste comma separated list of URLs").replace(/ /g, '');
      let encoded = encodeURIComponent(uris);
      location.hash = encoded;
    }
  } else if (ev.code == "KeyJ") {
    if (ev.ctrlKey) {
     exportToJson();
    }
  }
});
function process_split(){
  player.split();
}
function videoPlay(){
  if (player.playing) {
    player.pause();
    document.getElementById('videoplay').classList.add('mdi-arrow-right-drop-circle-outline');
    document.getElementById('videoplay').classList.remove('mdi-pause-circle-outline');
  } else {
    player.play();
    document.getElementById('videoplay').classList.remove('mdi-arrow-right-drop-circle-outline');
    document.getElementById('videoplay').classList.add('mdi-pause-circle-outline');
  }
}

function popup(text) {
  let popup = document.createElement('div');
  popup.addEventListener('keydown', function(ev) {
    ev.stopPropagation();
  });
  const close = document.createElement('a');
  close.addEventListener('click', function() {
    popup.remove();
  });
  close.textContent = "[x]";
  close.classList.toggle('close');
  popup.appendChild(close);
  popup.appendChild(text);
  popup.classList.toggle('popup');
  document.body.appendChild(popup);
  return popup;
}

window.addEventListener('load', function() {
  // traffic public here: https://jott.live/stat?path=/raw/mebm_hit
  var xhr = new XMLHttpRequest();
  let url = "https://jott.live/raw/mebm_hit";
  xhr.open("GET", url, true);
  xhr.send(null);
  
  // fix mobile touch
  document.getElementById('layer_holder').addEventListener("touchmove", function (e) {
    e.stopPropagation();
    //e.preventDefault();
  }, { passive: false });
  document.getElementById('export').addEventListener('click', download);

  if (location.hash) {
    let l = decodeURIComponent(location.hash.substring(1));
    for (let uri of l.split(',')) {
      player.addURI(uri);
    }
    location.hash = "";
    return;
  }
  let localStorage = window.localStorage;
  let seen = localStorage.getItem('_seen');
  if (!seen || false) {
    const text = document.createElement('div');
    text.innerHTML = `welcome!
      <br>
      <br>
      to start, drag in or paste URLs to videos and images.
      <br>
      a demo can be found <a href="https://bwasti.github.io/mebm/#https%3A%2F%2Fjott.live%2Fraw%2Ftutorial.json" target="_blank">here</a>
      and usage information <a href="https://github.com/bwasti/mebm#usage" target="_blank">here</a>.
      `;
    popup(text);
    localStorage.setItem('_seen', 'true');
  }

});

window.onbeforeunload = function() {
  return true;
}

window.addEventListener('resize', function() {
  player.resize();
});

window.addEventListener("touchmove", function (e) {
  e.preventDefault();
}, { passive: false });

function add_text() {
  let t = prompt("enter text");
  if (t) {
    player.add(new TextLayer(t));
  }
}

function exportVideo(blob) {
  alert("Warning: exported video may need to be fixed with cloudconvert.com or similar tools");
  const vid = document.createElement('video');
  vid.controls = true;
  vid.src = URL.createObjectURL(blob);
  backgroundElem(vid);
  let extension = blob.type.split(';')[0].split('/')[1];

  function make_a() {
    let h = document.getElementById('header');
    let a = h.querySelector('#download');
    if (!a) {
      a = document.createElement('a');
      a.id = 'download';
      a.download = (new Date()).getTime() + '.' + extension;
      a.textContent = 'download';
    }
    a.href = vid.src;
    document.getElementById('header').appendChild(a);
  }
  vid.ontimeupdate = function() {
    this.ontimeupdate = () => {
      return;
    }
    make_a();
    vid.currentTime = 0;
  }
  make_a();
  vid.currentTime = Number.MAX_SAFE_INTEGER;
}

function uploadSupportedType(files) {

  let badUserExtensions = [];

  for (let file of files) {
    let extension = file.name.split('.').pop();
    if (!(extension in ext_map)) {
      badUserExtensions.push(file)
    }
  }

  if (badUserExtensions.length) {
    const badFiles = badUserExtensions.map((ext)=>"- "+ext.name).join('<br>');
    const text = document.createElement('div');
    text.style.textAlign = "left";
    text.innerHTML = `
    the file(s) you uploaded are not supported :
    <br>
    <br>
    ${badFiles}
    `;
    popup(text);
  }
  return !badUserExtensions.length > 0;
}

function upload() {
  let f = document.getElementById('filepicker');
  f.addEventListener('input', function(e) {
    if(!uploadSupportedType(e.target.files)){return}
    for (let file of e.target.files) {
      player.addFile(file);
    }
    f.value = '';
  });
  f.click();
}

function getSupportedMimeTypes() {
  const VIDEO_TYPES = [
    "webm",
    "ogg",
    "mp4",
    "x-matroska"
  ];
  const VIDEO_CODECS = [
    "vp9",
    "vp9.0",
    "vp8",
    "vp8.0",
    "avc1",
    "av1",
    "h265",
    "h.265",
    "h264",
    "h.264",
    "opus",
  ];

  const supportedTypes = [];
  VIDEO_TYPES.forEach((videoType) => {
    const type = `video/${videoType}`;
    VIDEO_CODECS.forEach((codec) => {
      const variations = [
        `${type};codecs=${codec}`,
        `${type};codecs:${codec}`,
        `${type};codecs=${codec.toUpperCase()}`,
        `${type};codecs:${codec.toUpperCase()}`
      ]
      variations.forEach(variation => {
        if (MediaRecorder.isTypeSupported(variation))
          supportedTypes.push(variation);
      })
    });
    if (MediaRecorder.isTypeSupported(type)) supportedTypes.push(type);
  });
  return supportedTypes;
}

function download(ev) {
  // if (ev.shiftKey) {
  //   exportToJson();
  //   return;
  // }
  if (player.layers.length == 0) {
    alert("nothing to export");
    return;
  }

  const e = document.getElementById('export');
  const e_text = e.textContent;
  e.textContent = "exporting...";
  const chunks = [];
  const stream = player.canvas.captureStream();

  let has_audio = false;
  for (let layer of player.layers) {
    if (layer instanceof AudioLayer) {
      has_audio = true;
      break;
    }
  }

  if (has_audio) {
    let dest = player.audio_ctx.createMediaStreamDestination();
    player.audio_dest = dest;
    let tracks = dest.stream.getAudioTracks();
    stream.addTrack(tracks[0]);
  }

  const rec = new MediaRecorder(stream);
  rec.ondataavailable = e => chunks.push(e.data);

  rec.onstop = () => {
    const blob = new Blob(chunks, { type: "video/mp4" });
    uploadToServer(blob);
    e.textContent = e_text;
  };

  player.pause();
  player.time = 0;
  player.play();
  rec.start();

  player.onend(() => {
    rec.stop();
    player.audio_dest = null;
    player.pause();
    player.time = 0;
  });
  // let newTab = window.open();
  // newTab.document.write("<pre>" + JSON.stringify({
  //     note: player.dumpToJson()
  //   }, null, 2) + "</pre>");
  // newTab.document.close();
}

// Function to send the video to the server
function uploadToServer(blob) {
  const formData = new FormData();
  formData.append("video", blob, "edited-video.mp4");

  fetch("http://localhost:8080/upload", {
    method: "POST",
    body: formData,
  })
    .then(response => response.json())
    .then(data => {
      alert("Video uploaded successfully");

      // Show the download link to the user
      const downloadLink = document.getElementById('src');
      downloadLink.href = data.downloadUrl;
      downloadLink.textContent = "Click here to download the video";
      // document.body.appendChild(downloadLink);
    })
    .catch(error => {
      console.error("Error uploading video:", error);
      alert("Failed to upload video.");
    });
}



function uploadAndProcessVideo(filterType) {
  const formData = new FormData();

  if (!player.selected_layer) {
    console.log("No layer selected.");
    return;
  }
  console.log("player."+player);
  console.log("selected_layere."+player.selected_layer);

  // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).
  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer ."+file);
  if (!file) {
    console.log("Selected layer does not have a file.");
    return;
  }

  formData.append("file", file);

  // Send the file to the server for processing
  fetch(`http://localhost:8080/${filterType}`, {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (!response.ok) {
        throw new Error(`HTTP Error! Status: ${response.status}`);
    }
    return response.json();
})
  .then(data => {
    if (data.outputFile) {
      // Display the processed video on the canvas
      // displayProcessedVideo(data.outputFile);
      alert('file.uri'+data.nameFile);
      urlToFile(data.outputFile, data.nameFile, "video/mp4")
      .then((file) => {
        console.log("Converted file:", file); // Debugging
        file.uri = data.nameFile;
        alert('file.uri'+file.uri);
        player.addFile(file,true);
      })
      .catch((error) => console.error("Error converting URL to file:", error));
    } else {
      console.error('Error processing video:', data.error);
    }
  })
  .catch(error => {
    console.error('Error:', error);
  });
}

async function urlToFile(url, filename, mimeType) {
  try {
    const fullUrl = `http://localhost:8080/${url}`; // Ensure correct path
    console.log("Fetching file from:", fullUrl);
    
    const response = await fetch(fullUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch file: ${response.status} ${response.statusText}`);
    }
    
    const blob = await response.blob();
    return new File([blob], filename, { type: mimeType });
  } catch (error) {
    console.error("Error in urlToFile:", error);
    return null; // Return null instead of undefined
  }
}

function uploadAndProcessVideotranscribe(src,clusters) {
  const formData = new FormData();

  if (!player.selected_layer) {
    console.log("No layer selected.");
    const text = document.createElement('div');
    text.innerHTML="<p>No layer selected.</p>"
    popup(text);
    return;
  }
  console.log("player."+player);
  console.log("selected_layere."+player.selected_layer);

  // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).
  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer ."+file);
  if (!file) {
    console.log("Selected layer does not have a file.");
    const text = document.createElement('div');
    text.innerHTML="<p>you need to select video.</p>"
    popup(text);
    return;
  }
  const text = document.createElement('div');
  text.innerHTML="<p>Loading ... </p>"
  popup(text);
  formData.append("file", file);
  formData.append("src", src);
  formData.append("clusters", clusters);

  // Send the file to the server for processing
  fetch(`http://localhost:8080/process`, {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (!response.ok) {
        throw new Error(`HTTP Error! Status: ${response.status}`);
    }
    return response.json();
})
.then(data => {
  if (data.outputFile) {
    // Display the processed video on the canvas
    // displayProcessedVideo(data.outputFile);
    console.log("Converted data:", data); // Debugging
    urlToFile(data.outputFile, data.nameFile, "video/mp4")
    .then((file) => {
      console.log("Converted file:", file); // Debugging
      file.uri = data.nameFile;
      player.addFile(file,true);
    })
    .catch((error) => console.error("Error converting URL to file:", error));
    console.log("Converted file:", data.outputFile); 
    // downloadProcessedFile(data.outputFile, data.nameFile);
  } else {
    console.error('Error processing video:', data.error);
  }
})
  .catch(error => {
    console.error('Error:', error);
  });
}

function savetojson() {
  const filename = "data.json";
  const jsonStr = JSON.stringify(player.dumpToJson(), null, 2);
  const blob = new Blob([player.dumpToJson()], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function uploadSupportedTypejson(files) {
  return files.length > 0 && files[0].type === "application/json";
}

function getfromjson() {
  let f = document.getElementById("filepicker");
  f.addEventListener("input", function (e) {
      if (!uploadSupportedTypejson(e.target.files)) {
          console.error("Invalid file type. Please select a JSON file.");
          return;
      }

      for (let file of e.target.files) {
          let reader = new FileReader();
          reader.onload = function (event) {
              try {
                  const jsonData = JSON.parse(event.target.result);
                  console.log("Imported JSON:", jsonData);
                  player.loadFromJson(jsonData);
              } catch (error) {
                  console.error("Error parsing JSON:", error);
              }
          };
          reader.readAsText(file);
      }
      f.value = ""; // Reset input
  });

  f.click(); // Trigger file picker
}

 function loadproject(data) {
  player.removeall();
  console.log("Imported JSON:", data);
   const datajson = JSON.parse(data);
  player.loadFromJson(datajson);

  forceCloseModal('projectsModal');

}
function denoise(src,clusters) {
  const formData = new FormData();

  if (!player.selected_layer) {
    console.log("No layer selected.");
    const text = document.createElement('div');
    text.innerHTML="<p>No layer selected.</p>"
    popup(text);
    return;
  }
  console.log("player."+player);
  console.log("selected_layere."+player.selected_layer);

  // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).
  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer ."+file);
  if (!file) {
    console.log("Selected layer does not have a file.");
    const text = document.createElement('div');
    text.innerHTML="<p>you need to select video.</p>"
    popup(text);
    return;
  }
  const text = document.createElement('div');
  text.innerHTML="<p>Loading ... </p>"
  popup(text);
  formData.append("file", file);
  // Send the file to the server for processing
  fetch(`http://localhost:8080/denoise`, {
    method: 'POST',
    body: formData
})
.then(response => {
    if (!response.ok) {
        throw new Error(`HTTP Error! Status: ${response.status}`);
    }
    return response.json();
})
.then(data => {
    if (data.outputFile) {
        console.log("Converted data:", data);
        fetchAndDecodeAudio(data.outputFile, data.nameFile, "audio/wav")
        .then((file) => {
          console.log("Convertedfile.type file:", file.type); // Debugging
          var layer = player.addFile(file);
          file.uri = data.nameFile;
          layer.update_url(file.uri);
        
        })
        .catch((error) => console.error("Error converting URL to file:", error));
        

    } else {
        console.error('Error processing video:', data.error);
    }
})
.catch(error => {
    console.error('Error:', error);
});

}

  async function fetchAndDecodeAudio(url, filename, mimeType = "audio/wav") {
    try {
      const fullUrl = `http://localhost:8080/${url}`; // Ensure correct path
      console.log("Fetching file from:", fullUrl);
      
      const response = await fetch(fullUrl);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch file: ${response.status} ${response.statusText}`);
      }
  
      // Ensure response is in correct format
      const contentType = 'audio/wav';
      console.log("Response MIME Type:", contentType);
      
      if (!contentType || !contentType.includes("audio/wav")) {
        throw new Error(`Invalid MIME type: expected audio/wav, got ${contentType}`);
      }
  
      // Fetch as ArrayBuffer to prevent encoding issues
      const arrayBuffer = await response.arrayBuffer();
  
      // Convert ArrayBuffer to Blob
      const blob = new Blob([arrayBuffer], { type: mimeType });
  
      // Return a valid File object
      return new File([blob], filename, { type: mimeType });
      console.log("Convertedfile.type file:", file.type); // Debugging

    } catch (error) {
      console.error("Error in urlToFile:", error);
      return null; // Return null instead of undefined
    }
  }
  ////////////////////////////////////
  function startbuttonprocess() {   
    console.log("Processing start-button...");
    fetch("/start-processing", { method: "POST" }) // Request Node.js to start Python
    .then(response => {
        if (response.ok) {
          var result=  startWebSocket('process_data',''); // Start listening to progress updates
        } else {
            console.error("Failed to start processing");
        }
    })
    .catch(error => console.error("Error:", error));
}
function addnewlayerfromserver(reqdatajson) {
  const data= JSON.parse(reqdatajson);
  console.log("Converted outputFile:","/backend/"+data.outputFile); // Debugging
  urlToFile("backend/"+data.outputFile, data.fileName, "video/mp4")
  .then((file) => {
    console.log("Converted file:", file); // Debugging
    file.uri = file.name;
    player.addFile(file,true);
  })
  .catch((error) => console.error("Error converting URL to file:", error));
  console.log("Converted file:", data.outputFile); 

}

function updateinterface(data) {
  document.getElementById("progress-bar").style.width = data.progress + "%";
  document.getElementById("progress-text").innerText = 
      `Progress: ${data.progress}% | Time Left: ${data.time_left} sec`;
}

function startWebSocket(functiontitle,reqdata) {
  var ProcessingeModal = new bootstrap.Modal(document.getElementById("ProcessingeModal"));
  ProcessingeModal.show();
  const reqdatajson = JSON.parse(reqdata);
  console.log(" reqdata:", reqdata); // Debugging
  if (socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ function:functiontitle,data:reqdata }));
    console.log("📤 Sent message:", reqdata);

  } else {
    console.log("⚠️ WebSocket not connected.");
}

}

/////////////////////////////////
function transcribe_submit() {

   input_clusters = document.getElementById("input_clusters").value;
   input_src = document.getElementById("input_src").value;
  
  if (!player.selected_layer) {
    console.log("No layer selected.");
    const text = document.createElement('div');
    text.innerHTML="<p>No layer selected.</p>"
    popup(text);
    return;
  }
  console.log("player."+player);
  console.log("selected_layere."+player.selected_layer);

  // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).
  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer ."+file);
  if (!file) {
    console.log("Selected layer does not have a file.");
    const text = document.createElement('div');
    text.innerHTML="<p>you need to select video.</p>"
    popup(text);
    return;
  }
  const randomNum = Math.floor(Math.random() * 900000) + 100000;
  const fileName = `video_${randomNum}.mp4`;
  const inputFile = `uploads/${file}`;
  const outputFile = `uploads/${fileName}`;
  let data=JSON.stringify({ 
    clusters: input_clusters,
    src:input_src,
    inputFile:inputFile,
    fileName:fileName,
    outputFile:outputFile
  });
  console.log("fileName:", fileName); // Debugging

  forceCloseModal('transcribeModal');
  console.log("Processing start-button...");
  startWebSocket('transcribe',data);

  }

  function forceCloseModal(model) {
    var modalElement = document.getElementById(model);

    // Remove Bootstrap's "show" class
    modalElement.classList.remove("show");
    modalElement.style.display = "none";  // Hide modal

    // Remove the modal backdrop (dark overlay)
    var modalBackdrop = document.querySelector(".modal-backdrop");
    if (modalBackdrop) {
        modalBackdrop.remove();
    }

    // Remove "modal-open" class from <body> to allow scrolling
    document.body.classList.remove("modal-open");
}
////////////////////////////////////////////
function translate_submit(lang) {


  if (!player.selected_layer) {
    console.log("No layer selected.");
    const text = document.createElement('div');
    text.innerHTML="<p>No layer selected.</p>"
    popup(text);
    return;
  }
  console.log("player."+player);
  console.log("selected_layere."+player.selected_layer);

  // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).
  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer ."+file);
  if (!file) {
    console.log("Selected layer does not have a file.");
    const text = document.createElement('div');
    text.innerHTML="<p>you need to select video.</p>"
    popup(text);
    return;
  }
  const randomNum = Math.floor(Math.random() * 900000) + 100000;
  const fileName = `video_${randomNum}.mp4`;
  const inputFile = `uploads/${file}`;
  const outputFile = `uploads/${fileName}`;
  let data=JSON.stringify({ 
    clusters: input_clusters,
    src:input_src,
    lang:lang,
    inputFile:inputFile,
    fileName:fileName,
    outputFile:outputFile
  });
  console.log("fileName:", fileName); // Debugging
  startWebSocket('translate',data);
  // forceCloseModal('transcribeModal');
  }
  function process_data_submit() {


    // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).


    let data=JSON.stringify({ 

    });
  
   startWebSocket('process_data',data);
    }
 
    function applyBlurFilter() {
      switchTab('styles');
      document.querySelectorAll(".styles").forEach(el => el.style.display = "none");
      document.getElementById("blur").style.display = "block";
      let filtersigma = document.getElementById("filtersigma").value;
      let filteriterations = document.getElementById("filteriterations").value;
      let filterhorizontal_radius = document.getElementById("filterhorizontal_radius").value;
      let filtervertical_radius = document.getElementById("filtervertical_radius").value;

  if (!player.selected_layer) {
    console.log("No layer selected.");
    const text = document.createElement('div');
    text.innerHTML="<p>No layer selected.</p>"
    popup(text);
    return;
  }
  console.log("player."+player);
  console.log("selected_layere."+player.selected_layer);

  // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).
  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer ."+file);
  if (!file) {
    console.log("Selected layer does not have a file.");
    const text = document.createElement('div');
    text.innerHTML="<p>you need to select video.</p>"
    popup(text);
    return;
  }
  const randomNum = Math.floor(Math.random() * 900000) + 100000;
  const fileName = `video_${randomNum}.mp4`;
  const inputFile = `uploads/${file}`;
  const outputFile = `uploads/${fileName}`;
  let data=JSON.stringify({ 
    inputFile:inputFile,
    fileName:fileName,
    fps:fps,
    outputFile:outputFile,
    sigma:filtersigma,
    iterations:filteriterations,
    horizontal_radius:filterhorizontal_radius,
    vertical_radius:filtervertical_radius
  });
  socket.send(JSON.stringify({ function:'applyBlur',data:data }));
  }
  function adjustFilter(value) {
    console.log("Filter filtersigma changed:", value);
}
function applyBrightnessFilter() {
  switchTab('styles');
  document.querySelectorAll(".styles").forEach(el => el.style.display = "none");
  document.getElementById("brightness").style.display = "block";
  let filterbrightness = document.getElementById("filterbrightness").value;

  if (!player.selected_layer) {
    console.log("No layer selected.");
    const text = document.createElement('div');
    text.innerHTML="<p>No layer selected.</p>"
    popup(text);
    return;
  }
  console.log("player."+player);
  console.log("selected_layere."+player.selected_layer);

  // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).
  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer ."+file);
  if (!file) {
    console.log("Selected layer does not have a file.");
    const text = document.createElement('div');
    text.innerHTML="<p>you need to select video.</p>"
    popup(text);
    return;
  }
  const randomNum = Math.floor(Math.random() * 900000) + 100000;
  const fileName = `video_${randomNum}.mp4`;
  const inputFile = `uploads/${file}`;
  const outputFile = `uploads/${fileName}`;
  let data=JSON.stringify({ 
    inputFile:inputFile,
    fileName:fileName,
    fps:fps,
    outputFile:outputFile,
    brightness:filterbrightness
  });
  socket.send(JSON.stringify({ function:'applyBrightness',data:data }));
}
function applyContrastFilter() {
  switchTab('styles');
  document.querySelectorAll(".styles").forEach(el => el.style.display = "none");
  document.getElementById("contrast").style.display = "block";
  let filtercontrast = document.getElementById("filtercontrast").value;
  
  if (!player.selected_layer) {
    console.log("No layer selected.");
    const text = document.createElement('div');
    text.innerHTML="<p>No layer selected.</p>"
    popup(text);
    return;
  }
  console.log("player."+player);
  console.log("selected_layere."+player.selected_layer);

  // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).
  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer ."+file);
  if (!file) {
    console.log("Selected layer does not have a file.");
    const text = document.createElement('div');
    text.innerHTML="<p>you need to select video.</p>"
    popup(text);
    return;
  }
  const randomNum = Math.floor(Math.random() * 900000) + 100000;
  const fileName = `video_${randomNum}.mp4`;
  const inputFile = `uploads/${file}`;
  const outputFile = `uploads/${fileName}`;
  let data=JSON.stringify({ 
    inputFile:inputFile,
    fileName:fileName,
    fps:fps,
    outputFile:outputFile,
    contrast:filtercontrast
  });
  socket.send(JSON.stringify({ function:'applyContrast',data:data }));
}
function applyGrayscaleFilter() {
  switchTab('styles');
  document.querySelectorAll(".styles").forEach(el => el.style.display = "none");
  document.getElementById("grayscale").style.display = "block";
  
  if (!player.selected_layer) {
    console.log("No layer selected.");
    const text = document.createElement('div');
    text.innerHTML="<p>No layer selected.</p>"
    popup(text);
    return;
  }
  console.log("player."+player);
  console.log("selected_layere."+player.selected_layer);

  // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).
  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer ."+file);
  if (!file) {
    console.log("Selected layer does not have a file.");
    const text = document.createElement('div');
    text.innerHTML="<p>you need to select video.</p>"
    popup(text);
    return;
  }
  const randomNum = Math.floor(Math.random() * 900000) + 100000;
  const fileName = `video_${randomNum}.mp4`;
  const inputFile = `uploads/${file}`;
  const outputFile = `uploads/${fileName}`;
  let data=JSON.stringify({ 
    inputFile:inputFile,
    fileName:fileName,
    fps:fps,
    outputFile:outputFile
   
  });
  socket.send(JSON.stringify({ function:'applyGrayscale',data:data }));
}
function applySepiaFilter() {
  switchTab('styles');
  document.querySelectorAll(".styles").forEach(el => el.style.display = "none");
  document.getElementById("sepia").style.display = "block";
  
  if (!player.selected_layer) {
    console.log("No layer selected.");
    const text = document.createElement('div');
    text.innerHTML="<p>No layer selected.</p>"
    popup(text);
    return;
  }
  console.log("player."+player);
  console.log("selected_layere."+player.selected_layer);

  // Assuming selected_layer has a 'file' or 'uri' property (you can customize based on your class).
  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer ."+file);
  if (!file) {
    console.log("Selected layer does not have a file.");
    const text = document.createElement('div');
    text.innerHTML="<p>you need to select video.</p>"
    popup(text);
    return;
  }
  const randomNum = Math.floor(Math.random() * 900000) + 100000;
  const fileName = `video_${randomNum}.mp4`;
  const inputFile = `uploads/${file}`;
  const outputFile = `uploads/${fileName}`;
  let data=JSON.stringify({ 
    inputFile:inputFile,
    fileName:fileName,
    fps:fps,
    outputFile:outputFile
   
  });
  socket.send(JSON.stringify({ function:'applySepia',data:data }));
}
function renderFinalVideo() {
  if (!player.selected_layer) {
      console.log("No layer selected.");
      const text = document.createElement('div');
      text.innerHTML = "<p>No layer selected.</p>";
      popup(text);
      return;
  }

  let file = player.selected_layer.getSelectedLayerFile(); // Get the file from the selected layer
  console.log("Selected layer:", file);
  
  if (!file) {
      console.log("Selected layer does not have a file.");
      const text = document.createElement('div');
      text.innerHTML = "<p>You need to select a video.</p>";
      popup(text);
      return;
  }

  const randomNum = Math.floor(Math.random() * 900000) + 100000;
  const fileName = `video_${randomNum}.mp4`;
  const inputFile = `uploads/${file}`; 
  const outputFile = `uploads/${fileName}`;

  let renderData = JSON.stringify({
      inputFile: inputFile,
      fileName: fileName,
      fps: fps,
      outputFile: outputFile
  });

  console.log("🔹 Sending render request:", renderData);

  socket.send(JSON.stringify({ function: 'renderVideo', data: renderData }));
}
async function saveproject() {
  const name = document.getElementById('projectName').value;
  const data = JSON.stringify(player.dumpToJson(), null, 2);
  const token = localStorage.getItem('token');
  const userId = localStorage.getItem('userId');
  
  const res = await fetch('http://localhost:8080/api/projects', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify({ userId, name, data })
  });

  const result = await res.json();
  if (res.ok) alert('Project saved successfully!');
  else alert('Failed to save project');
}
function logout() {
  localStorage.removeItem('token');
  localStorage.removeItem('userId');
  window.location.href = '/login.html';
}