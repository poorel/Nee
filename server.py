"""中文评论负面片段抽取服务: 提交评论 -> token头(BIO)圈出负面片段"""
import json, os, re
from http.server import HTTPServer, BaseHTTPRequestHandler
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL = os.environ.get("MODEL_PATH", "./model")
PORT = int(os.environ.get("PORT", "8910"))
LABELS = ["O", "B-NEG", "I-NEG"]
MAXLEN = 256   # v10 起位置编码训到 256, 服务端同步放开(v9 时代为 160)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForTokenClassification.from_pretrained(MODEL)
model.eval()

def strip_punct(s):
    return re.sub(r'^[。．，,、！!？?~～\.…\s]+|[。．，,、！!？?~～\.…\s]+$', '', s)

def extract(text):
    enc = tok(list(text), is_split_into_words=True, truncation=True,
              max_length=MAXLEN, return_tensors="pt")
    wids = enc.word_ids(0)
    with torch.no_grad():
        logits = model(**enc).logits[0]
        probs = torch.softmax(logits, -1)
        pred = probs.argmax(-1).tolist()
    char_tags, char_bprob = {}, {}
    for j, w in enumerate(wids):
        if w is not None and w < len(text) and w not in char_tags:
            char_tags[w] = LABELS[pred[j]]
            char_bprob[w] = float(probs[j][1])  # B-NEG 置信度
    spans, cur = [], None
    # I-run 无 B 起点修复(v10 验收过线关键): 连续 I-NEG(段长>=2)缺 B 时首字视为 B
    prev = "O"
    for i in range(len(text)):
        ti = char_tags.get(i, "O")
        if ti == "I-NEG" and prev == "O" and char_tags.get(i+1, "O") == "I-NEG":
            char_tags[i] = "B-NEG"
        prev = ti
    for i in range(len(text)):
        t = char_tags.get(i, "O")
        if t == "B-NEG":
            if cur: spans.append(cur)
            cur = [i, i + 1]
        elif t == "I-NEG" and cur:
            cur[1] = i + 1
        else:
            if cur: spans.append(cur); cur = None
    if cur: spans.append(cur)
    out = []
    for s, e in spans:
        # 护栏: 单字 span 且起始字 B 置信度 < 0.7 视为临界误点火, 丢弃
        if e - s == 1 and char_bprob.get(s, 0.0) < 0.7:
            continue
        frag = strip_punct(text[s:e])
        if frag:
            out.append({"span": frag, "start": text.index(frag, max(0, s-3)) if frag in text else s})
    return out

WINDOW = 140   # 滑窗长度(与训练 MAXLEN=160 对齐留余量)

def extract_long(text):
    """长文按句子边界打包窗口推理: 负面段尽量靠窗前, 避免被截断+减少正面铺垫压制"""
    if len(text) <= WINDOW:
        return extract(text)
    import re as _re
    # 按 。！？ 分句(保留句尾标点), 子句不切断
    sents = _re.findall(r'[^。！？]*[。！？]|[^。！？]+$', text)
    chunks, cur = [], ""
    for s in sents:
        if cur and len(cur) + len(s) > WINDOW:
            chunks.append(cur)
            cur = s
        else:
            cur += s
    if cur:
        chunks.append(cur)
    seen, out = set(), []
    for ci, chunk in enumerate(chunks):
        for d in extract(chunk):
            if d["span"] not in seen:
                seen.add(d["span"])
                out.append({"span": d["span"], "start": chunk_start_offset(ci, chunks, d)})
    return out

def chunk_start_offset(ci, chunks, d):
    # 窗口内相对 start 累加为全文绝对偏移
    off = sum(len(c) for c in chunks[:ci])
    return off + d["start"]

PAGE = """<!doctype html><html lang=zh><head><meta charset=utf-8>
<title>负面片段抽取测试</title><style>
body{font-family:system-ui;max-width:760px;margin:40px auto;padding:0 16px;background:#fafafa}
h2{color:#333} textarea{width:100%;height:90px;font-size:16px;padding:10px}
button{font-size:16px;padding:8px 24px;background:#c0392b;color:#fff;border:0;border-radius:6px;cursor:pointer}
.result{margin-top:20px;padding:14px;border:1px solid #ddd;border-radius:8px;background:#fff;font-size:16px;line-height:1.8}
.neg{background:#ffe0b2;border-bottom:2px solid #e67e22;padding:0 2px;border-radius:3px}
.empty{color:#27ae60}.note{color:#888;font-size:13px;margin-top:8px}
ul{margin:8px 0 0;padding-left:20px} li{margin:2px 0}
</style></head><body>
<h2>负面评价片段抽取 (macbert_token_neg_v11, 验收五道全过)</h2>
<textarea id=txt placeholder=输入一条评论,如: 物流太慢了,包装倒是完好,但客服态度很差></textarea><br><br>
<button onclick=go()>抽取负面片段</button>
<div id=out></div>
<script>
async function go(){
  const r = await fetch('/api/extract',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({text:document.getElementById('txt').value})});
  const d = await r.json();
  let html = '<div class=result><b>原文:</b><br>' + d.highlighted;
  if(d.spans.length){ html += '<br><b>圈中 ' + d.spans.length + ' 个片段:</b><ul>';
    d.spans.forEach(s=>html+='<li class=neg>'+s+'</li>'); html+='</ul>'; }
  else html += '<br><b class=empty>未圈出负面片段</b>';
  html += '<div class=note>橙色=模型抽取的负面片段; 首尾标点已自动剥离</div></div>';
  document.getElementById('out').innerHTML = html;
}
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self._send(200, PAGE.encode(), "text/html; charset=utf-8")
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        text = json.loads(self.rfile.read(n)).get("text", "").strip()
        if not text:
            self._send(400, b'{"error":"empty"}', "application/json"); return
        spans = extract_long(text)
        hl, last = [], 0
        for sp in spans:
            i = text.find(sp["span"], last)
            if i < 0: continue
            hl.append(_esc(text[last:i])); hl.append('<span class=neg>' + _esc(sp["span"]) + '</span>')
            last = i + len(sp["span"])
        hl.append(_esc(text[last:]))
        body = json.dumps({"spans": [s["span"] for s in spans],
                           "highlighted": "".join(hl)}, ensure_ascii=False).encode()
        self._send(200, body, "application/json; charset=utf-8")
    def log_message(self, *a): pass

def _esc(s):
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

if __name__ == "__main__":
    print(f"serving on 0.0.0.0:{PORT}, model={MODEL}", flush=True)
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()
