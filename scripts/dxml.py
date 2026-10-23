# -*- coding: utf-8 -*-
"""XML-level docx text editor that respects tracked changes.

python-docx cannot see runs nested inside <w:ins>/<w:del>, so Mahdi's tracked
edits are invisible to it. This walks every <w:t> in document order, treating
accepted-final text as: all w:t (including those inside w:ins), excluding
w:delText (his deletions). Replacements are applied across element boundaries.
"""
import re, zipfile, shutil, os
import xml.etree.ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
ET.register_namespace("w", W[1:-1])

class Doc:
    def __init__(self, path):
        self.path = path
        self.zin = zipfile.ZipFile(path)
        self.xml = self.zin.read("word/document.xml").decode("utf-8")
        self.root = ET.fromstring(self.xml)

    def _texts(self):
        """Every w:t element in document order, skipping those inside w:del."""
        out = []
        for p in self.root.iter(W+"p"):
            for el in p.iter():
                if el.tag == W+"t":
                    # is this t inside a w:del? walk ancestors via a parent map
                    out.append((p, el))
        return out

    def build_index(self):
        """Flat final text + list of (element, start, end)."""
        parents = {c: pa for pa in self.root.iter() for c in pa}
        def in_del(el):
            cur = parents.get(el)
            while cur is not None:
                if cur.tag == W+"del": return True
                cur = parents.get(cur)
            return False
        self.spans = []
        buf = []
        pos = 0
        for p in self.root.iter(W+"p"):
            for el in p.iter(W+"t"):
                if in_del(el): continue
                t = el.text or ""
                self.spans.append((el, pos, pos+len(t)))
                buf.append(t); pos += len(t)
            self.spans.append((None, pos, pos+1)); buf.append("\n"); pos += 1
        self.flat = "".join(buf)
        return self.flat

    def replace(self, pattern, repl, label="", flags=re.S, count=1):
        """Regex replace over the flat text, writing back across elements."""
        done = 0
        for _ in range(count):
            self.build_index()
            m = re.search(pattern, self.flat, flags)
            if not m: break
            i, j = m.span()
            new = m.expand(repl) if "\\" in repl else repl
            touched = [(el,s,e) for el,s,e in self.spans if el is not None and e > i and s < j]
            if not touched: break
            first = touched[0]; last = touched[-1]
            pre = (first[0].text or "")[: i - first[1]]
            post = (last[0].text or "")[j - last[1]:] if last[2] > j else ""
            if first[0] is last[0]:
                first[0].text = pre + new + post
            else:
                first[0].text = pre + new
                for el,_,_ in touched[1:-1]: el.text = ""
                last[0].text = post
            done += 1
        return done

    def save(self, out):
        body = ET.tostring(self.root, encoding="unicode")
        shutil.copy(self.path, out) if self.path != out else None
        tmp = out + ".tmp"
        with zipfile.ZipFile(self.path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for it in zin.infolist():
                data = zin.read(it.filename)
                if it.filename == "word/document.xml":
                    data = body.encode("utf-8")
                zout.writestr(it, data)
        os.replace(tmp, out)
