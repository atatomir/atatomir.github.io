/* ScreenToVideo — site behaviors. Vanilla JS, loaded with `defer`. */
(function () {
  'use strict';

  // ---- Config -------------------------------------------------------------
  var ORIGIN = 'https://screentovideo.com';
  var CFG = {
    adsId: 'AW-17501610339',
    downloadConversion: 'AW-17501610339/yF1oCJy2gowbEOPCtZlB',
    tawk: 'https://embed.tawk.to/5e93399035bcbb0c9ab04034/default',
    tracker: ORIGIN + '/pages/page.php',
    installer: ORIGIN + '/svsetup.exe',
    support: 'support@screentovideo.com',
    // Set to a form backend URL (Formspree, Basin, your own PHP, ...) to submit
    // forms in-page. While empty, forms open the visitor's email app instead.
    formEndpoint: ''
  };
  var store = {
    PRODUCT_REG: 57852, PRODUCT_EDU: 61278, TEMPLATE: 13547,
    url: function (id, qty, coupon) {
      return 'https://store.payproglobal.com/checkout?products[1][id]=' + id +
        '&page-template=' + this.TEMPLATE + '&products[1][qty]=' + qty +
        (coupon ? '&coupon-code-to-add=' + encodeURIComponent(coupon) : '');
    }
  };

  var d = document, body = d.body, page = body.getAttribute('data-page') || '';
  var $ = function (s, c) { return (c || d).querySelector(s); };
  var $$ = function (s, c) { return Array.prototype.slice.call((c || d).querySelectorAll(s)); };
  var reduced = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
  var qs = new URLSearchParams(location.search);

  function setCookie(n, v, days) {
    var t = new Date(); t.setTime(t.getTime() + days * 864e5);
    d.cookie = n + '=' + encodeURIComponent(v) + ';expires=' + t.toUTCString() + ';path=/;SameSite=Lax';
  }
  function getCookie(n) {
    var m = d.cookie.match(new RegExp('(?:^|; )' + n + '=([^;]*)'));
    return m ? decodeURIComponent(m[1]) : '';
  }
  function ls(k, v) {
    try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; }
  }
  function idle(fn, ms) {
    var done = false, run = function () { if (!done) { done = true; fn(); } };
    var start = function () { setTimeout(function () { (window.requestIdleCallback || setTimeout)(run); }, ms); };
    if (d.readyState === 'complete') start(); else addEventListener('load', start, { once: true });
    ['pointerdown', 'keydown', 'scroll', 'touchstart'].forEach(function (e) { addEventListener(e, run, { once: true, passive: true }); });
  }

  // ---- Campaign coupon (?src=google|bing|betapage) -------------------------
  var src = (qs.get('src') || '').toLowerCase();
  if (src === 'google' || src === 'bing') setCookie('coupon', 'SV50OFF', 365);
  if (src === 'betapage') setCookie('coupon', 'SVP50OFF', 365);

  // ---- Page-view ping (existing server-side counter) ----------------------
  var track = body.getAttribute('data-track');
  if (track === 'uploadtogd') track += (qs.get('status') || '').toLowerCase();
  if (track) {
    try {
      fetch(CFG.tracker + '?page=' + encodeURIComponent(track) + '&referrer=' + encodeURIComponent(d.referrer),
        { mode: 'no-cors', keepalive: true, credentials: 'omit' }).catch(function () {});
    } catch (e) {}
  }

  // ---- Header -------------------------------------------------------------
  var hdr = $('.hdr');
  if (hdr) {
    var onScroll = function () { hdr.classList.toggle('scrolled', scrollY > 8); };
    onScroll(); addEventListener('scroll', onScroll, { passive: true });
    var mb = $('.menu-btn', hdr);
    if (mb) mb.addEventListener('click', function () {
      var open = hdr.classList.toggle('open');
      mb.setAttribute('aria-expanded', open);
      mb.querySelector('use').setAttribute('href', mb.getAttribute('data-icons') + (open ? '#close' : '#menu'));
    });
  }

  // ---- Typewriter ---------------------------------------------------------
  var typed = $('[data-words]');
  if (typed) {
    var words = typed.getAttribute('data-words').split('|');
    // Reserve the height of the tallest word, so a long word that wraps never pushes the page down
    var tline = typed.parentNode;
    var fitTyped = function () {
      // one hidden copy of the line per word, measured together (a single layout pass)
      var probe = d.createElement('div');
      probe.setAttribute('aria-hidden', 'true');
      probe.style.cssText = 'position:absolute;left:0;top:0;visibility:hidden;pointer-events:none;width:' + tline.getBoundingClientRect().width + 'px';
      probe.innerHTML = words.map(function (w) { return '<span class="typed-line" style="min-height:0">with <span class="typed">' + w.replace(/[&<>]/g, '') + '</span><span class="caret"></span></span>'; }).join('');
      tline.parentNode.appendChild(probe);
      var mx = 0; $$('.typed-line', probe).forEach(function (x) { mx = Math.max(mx, x.offsetHeight); });
      probe.remove();
      tline.style.minHeight = mx + 'px';
    };
    fitTyped();
    var tRz; addEventListener('resize', function () { clearTimeout(tRz); tRz = setTimeout(fitTyped, 100); });
    if (reduced) typed.textContent = words[0];
    else {
      var wi = 0, ci = 0, del = false;
      (function tick() {
        var w = words[wi];
        if (!del) {
          typed.textContent = w.slice(0, ++ci);
          if (ci === w.length) { del = true; return setTimeout(tick, 1600); }
          return setTimeout(tick, 55);
        }
        typed.textContent = w.slice(0, --ci);
        if (ci === 0) { del = false; wi = (wi + 1) % words.length; return setTimeout(tick, 220); }
        setTimeout(tick, 22);
      })();
    }
  }

  // ---- Reveal on scroll ---------------------------------------------------
  if ('IntersectionObserver' in window && !reduced) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
    }, { rootMargin: '0px 0px -8% 0px' });
    $$('.rv').forEach(function (el) { io.observe(el); });
  } else $$('.rv').forEach(function (el) { el.classList.add('in'); });

  // ---- Card spotlight -----------------------------------------------------
  if (matchMedia('(hover:hover)').matches) {
    d.addEventListener('pointermove', function (e) {
      var c = e.target.closest && e.target.closest('.card-spot');
      if (!c) return;
      var r = c.getBoundingClientRect();
      c.style.setProperty('--mx', (e.clientX - r.left) + 'px');
      c.style.setProperty('--my', (e.clientY - r.top) + 'px');
    }, { passive: true });
  }

  // ---- Video modal (videos load only on demand) ---------------------------
  var modal = $('#vmodal');
  if (modal) {
    var mv = $('video', modal), mt = $('h3', modal), navs = $$('.modal-nav', modal), list = [], cur = -1;
    function show(t) {
      mt.textContent = t.getAttribute('data-title') || '';
      mv.src = t.getAttribute('data-video');
      list = $$('.card-btn[data-video], .ai-demo[data-video]').filter(function (x) { return x.offsetParent !== null; });
      cur = list.indexOf(t);
      navs.forEach(function (n) { n.hidden = cur < 0 || list.length < 2; });
      var p = mv.play(); if (p && p.catch) p.catch(function () {});
    }
    function step(dir) { if (cur < 0) return; show(list[(cur + dir + list.length) % list.length]); }
    navs.forEach(function (n) { n.addEventListener('click', function () { step(+n.getAttribute('data-dir')); }); });
    modal.addEventListener('keydown', function (e) {
      if (e.target === mv && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) return; // let the player seek
      if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
      if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
    });
    d.addEventListener('click', function (e) {
      var t = e.target.closest && e.target.closest('[data-video]');
      if (!t || modal.contains(t)) return;
      e.preventDefault();
      if (modal.showModal) { if (!modal.open) modal.showModal(); } else modal.setAttribute('open', '');
      show(t);
    });
    var close = function () { mv.pause(); mv.removeAttribute('src'); mv.load(); if (modal.open) modal.close(); };
    $('.modal-close', modal).addEventListener('click', close);
    modal.addEventListener('click', function (e) { if (e.target === modal) close(); });
    modal.addEventListener('close', function () { mv.pause(); mv.removeAttribute('src'); mv.load(); });
  }

  // ---- Features sub-nav scrollspy ----------------------------------------
  var sn = $('.subnav');
  if (sn) {
    var bar = $('.subnav-links', sn) || sn, links = $$('a', bar), map = {}, lockUntil = 0;
    links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var setActive = function (a) {
      links.forEach(function (x) { x.classList.toggle('on', x === a); });
      // Scroll only the tab strip (never scrollIntoView: it would cancel the page's smooth scroll)
      if (a && bar.scrollWidth > bar.clientWidth) {
        var left = a.offsetLeft - (bar.clientWidth - a.offsetWidth) / 2;
        bar.scrollLeft = Math.max(0, left);
      }
    };
    links.forEach(function (a) {
      a.addEventListener('click', function (e) {
        var t = d.getElementById(a.getAttribute('href').slice(1));
        if (!t) return;
        e.preventDefault();
        var offset = (hdr ? hdr.offsetHeight : 0) + sn.offsetHeight + 8;
        var y = t.getBoundingClientRect().top + scrollY - offset;
        lockUntil = Date.now() + 1200;
        setActive(a);
        scrollTo({ top: y, behavior: reduced ? 'auto' : 'smooth' });
        if (history.replaceState) history.replaceState(null, '', '#' + t.id);
      });
    });
    if ('IntersectionObserver' in window) {
      var spy = new IntersectionObserver(function (es) {
        if (Date.now() < lockUntil) return;
        es.forEach(function (e) { if (e.isIntersecting) setActive(map[e.target.id]); });
      }, { rootMargin: '-45% 0px -50% 0px' });
      Object.keys(map).forEach(function (id) { var s = d.getElementById(id); if (s) spy.observe(s); });
    }
  }

  // ---- Changelog filters --------------------------------------------------
  var clfs = $$('.clf');
  if (clfs.length) {
    var items = $$('.cl-item'), ysecs = $$('.cl-ysec');
    clfs.forEach(function (b) {
      b.addEventListener('click', function () {
        var k = b.getAttribute('data-k');
        clfs.forEach(function (x) { var on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-pressed', on); });
        items.forEach(function (it) {
          var any = false;
          $$('li[data-k]', it).forEach(function (li) { var hit = k === 'all' || li.getAttribute('data-k') === k; li.hidden = !hit; any = any || hit; });
          var co = $('.callout', it); if (co) co.hidden = k !== 'all';
          it.hidden = !any;
          if (any) it.classList.add('in');
        });
        ysecs.forEach(function (y) { y.hidden = !$$('.cl-item:not([hidden])', y).length; });
      });
    });
  }

  // ---- "Who is it for": collapsed rows on phones, always open on larger screens --
  var audItems = $$('details.aud-item');
  if (audItems.length && window.matchMedia) {
    var audMq = matchMedia('(max-width: 620px)');
    var audSync = function () { audItems.forEach(function (x) { x.open = !audMq.matches; }); };
    audSync();
    if (audMq.addEventListener) audMq.addEventListener('change', audSync);
    audItems.forEach(function (x) { $('summary', x).addEventListener('click', function (ev) { if (!audMq.matches) ev.preventDefault(); }); });
  }

  // ---- Recording modes (home "Record your Screen" card) -------------------
  $$('.chips .chip[data-desc]').forEach(function (c) {
    c.addEventListener('click', function () {
      var box = c.parentNode, desc = c.closest('.card').querySelector('.rec-desc');
      $$('.chip', box).forEach(function (x) { var on = x === c; x.classList.toggle('on', on); x.setAttribute('aria-pressed', on); });
      if (!desc || desc.textContent === c.getAttribute('data-desc')) return;
      desc.classList.add('swap');
      setTimeout(function () { desc.textContent = c.getAttribute('data-desc'); desc.classList.remove('swap'); }, reduced ? 0 : 150);
    });
  });

  // ---- Reviews carousel: endless in both directions ----------------------
  // A hidden copy of the cards sits on each side; once a scroll settles in a copy,
  // we jump (invisibly) by one set width back to the real cards.
  var revs = $('.revs');
  if (revs) {
    var orig = $$('.rvw', revs);
    var copy = function () {
      return orig.map(function (c) {
        var x = c.cloneNode(true);
        x.setAttribute('aria-hidden', 'true'); x.removeAttribute('aria-label');
        $$('a', x).forEach(function (a) { a.tabIndex = -1; });
        return x;
      });
    };
    copy().forEach(function (x) { revs.insertBefore(x, orig[0]); });
    copy().forEach(function (x) { revs.appendChild(x); });
    var setW = function () { return orig[0].offsetLeft - revs.firstElementChild.offsetLeft; };
    var step = function () { return orig[1] ? orig[1].offsetLeft - orig[0].offsetLeft : setW(); };
    var jump = function (x) { revs.style.scrollSnapType = 'none'; revs.scrollTo({ left: x, behavior: 'instant' }); revs.style.scrollSnapType = ''; };
    var settle = function () {
      var W = setW(), x = revs.scrollLeft;
      if (x < W - 2) jump(x + W); else if (x >= 2 * W - 2) jump(x - W);
    };
    jump(setW());
    var idx = 0;
    addEventListener('resize', function () { jump(setW() + idx * step()); });
    var rT, onSettled = function () { settle(); idx = ((Math.round((revs.scrollLeft - setW()) / step()) % orig.length) + orig.length) % orig.length; };
    if ('onscrollend' in window) revs.addEventListener('scrollend', onSettled);
    else revs.addEventListener('scroll', function () { clearTimeout(rT); rT = setTimeout(onSettled, 150); }, { passive: true });
    var rGo = function (dir) { revs.scrollBy({ left: dir * (revs.clientWidth + 16), behavior: reduced ? 'auto' : 'smooth' }); };
    $('.rev-prev').addEventListener('click', function () { rGo(-1); });
    $('.rev-next').addEventListener('click', function () { rGo(1); });
    revs.addEventListener('keydown', function (ev) { if (ev.key === 'ArrowRight' || ev.key === 'ArrowLeft') { ev.preventDefault(); rGo(ev.key === 'ArrowRight' ? 1 : -1); } });
  }

  // ---- Help center: contents (collapsed on phones) + search ---------------
  var hnd = $('.hc-navd');
  if (hnd && matchMedia('(max-width: 960px)').matches) hnd.open = false;
  $$('.hc-search').forEach(function (box) {
    var inp = $('input', box), res = $('.hc-res', box), idx = null, sel = -1;
    var esc = function (s) { return s.replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
    var load = function () {
      if (idx) return Promise.resolve(idx);
      return fetch(box.getAttribute('data-index')).then(function (r) { return r.json(); }).then(function (d) {
        idx = d.map(function (x) { x.tl = x.t.toLowerCase(); x.xl = x.x.toLowerCase(); return x; }); return idx;
      });
    };
    var hl = function (s, words) {
      var out = esc(s);
      words.forEach(function (w) { out = out.replace(new RegExp('(' + w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>'); });
      return out;
    };
    var run = function () {
      var q = inp.value.trim().toLowerCase();
      if (q.length < 2) { res.hidden = true; return; }
      load().then(function (d) {
        var words = q.split(/\s+/).filter(Boolean);
        var hits = d.map(function (x) {
          var sc = 0;
          for (var i = 0; i < words.length; i++) {
            var w = words[i], inT = x.tl.indexOf(w) > -1, inX = x.xl.indexOf(w) > -1;
            if (!inT && !inX) return null;
            sc += (inT ? 10 : 0) + (inX ? 1 + Math.min(4, x.xl.split(w).length - 1) : 0);
          }
          if (x.tl.indexOf(q) === 0) sc += 20;
          return { x: x, sc: sc };
        }).filter(Boolean).sort(function (a, b) { return b.sc - a.sc; }).slice(0, 8);
        sel = -1;
        res.innerHTML = hits.length ? hits.map(function (h) {
          var x = h.x, p = x.xl.indexOf(words[0]), snip = '';
          if (p > -1) { var a = Math.max(0, p - 50); snip = (a ? '…' : '') + x.x.slice(a, a + 140) + '…'; }
          else snip = x.x.slice(0, 120) + '…';
          return '<a href="' + x.f + '" role="option">' + (x.s ? '<small>' + esc(x.s) + '</small>' : '') + '<b>' + hl(x.t, words) + '</b><span>' + hl(snip, words) + '</span></a>';
        }).join('') : '<p>No results for “' + esc(inp.value.trim()) + '”. Try another word, or <a href="../contact/" style="display:inline;padding:0;color:var(--teal-2)">contact support</a>.</p>';
        res.hidden = false;
      });
    };
    inp.addEventListener('focus', load, { once: true });
    inp.addEventListener('input', run);
    inp.addEventListener('keydown', function (ev) {
      var links = $$('a[role=option]', res);
      if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
        if (!links.length) return;
        ev.preventDefault();
        sel = (sel + (ev.key === 'ArrowDown' ? 1 : -1) + links.length) % links.length;
        links.forEach(function (l, i) { l.classList.toggle('on', i === sel); });
        links[sel].scrollIntoView({ block: 'nearest' });
      } else if (ev.key === 'Enter') {
        var go = links[sel > -1 ? sel : 0]; if (go) { ev.preventDefault(); location.href = go.href; }
      } else if (ev.key === 'Escape') { res.hidden = true; }
    });
    document.addEventListener('click', function (ev) { if (!box.contains(ev.target)) res.hidden = true; });
  });

  // ---- Back to top --------------------------------------------------------
  var tt = $('.totop'), cbar = $('.consent'), ftr = $('.ftr'), ftrIn = false;
  if (tt) {
    var onTT = function () { tt.classList.toggle('show', scrollY > 1400 && !ftrIn && !(cbar && cbar.classList.contains('show'))); };
    // Hide once the footer is on screen, so it never covers the footer links, copyright or version
    if (ftr && 'IntersectionObserver' in window) new IntersectionObserver(function (es) { ftrIn = es[0].isIntersecting; onTT(); }).observe(ftr);
    addEventListener('scroll', onTT, { passive: true }); onTT();
    tt.addEventListener('click', function () { scrollTo({ top: 0, behavior: reduced ? 'auto' : 'smooth' }); });
  }

  // ---- Feature search -----------------------------------------------------
  var fs = $('[data-fsearch]');
  if (fs) {
    var tiles = $$('.fgroup .card'), groups = $$('.fgroup'), empty = $('.fempty');
    var norm = function (x) { return x.toLowerCase().replace(/[^a-z0-9]+/g, ' '); };
    tiles.forEach(function (t) { t._q = norm(t.textContent); });
    var filter = function () {
      var q = norm(fs.value).trim(), any = false;
      tiles.forEach(function (t) { var hit = !q || q.split(' ').every(function (w) { return t._q.indexOf(w) > -1; }); t.classList.toggle('is-hidden', !hit); if (hit) t.classList.add('in'); });
      groups.forEach(function (g) { var vis = $$('.card:not(.is-hidden)', g).length; g.classList.toggle('is-hidden', !vis); any = any || vis > 0; });
      if (empty) empty.hidden = any;
    };
    fs.addEventListener('input', filter);
    fs.addEventListener('focus', function () { var f = $('#recording'); if (f && f.getBoundingClientRect().top > innerHeight) f.scrollIntoView(); });
    var clr = $('[data-fclear]'); if (clr) clr.addEventListener('click', function (e) { e.preventDefault(); fs.value = ''; filter(); fs.focus(); });
  }

  // ---- Instant navigation fallback (browsers without speculation rules) ---
  if (!(HTMLScriptElement.supports && HTMLScriptElement.supports('speculationrules'))) {
    var seen = {};
    var pf = function (e) {
      var a = e.target.closest && e.target.closest('a[href]');
      if (!a || a.target || a.origin !== location.origin || /\.exe$|#/.test(a.href) || seen[a.href] || a.href === location.href) return;
      seen[a.href] = 1;
      var l = d.createElement('link'); l.rel = 'prefetch'; l.href = a.href; d.head.appendChild(l);
    };
    d.addEventListener('pointerover', pf, { passive: true });
    d.addEventListener('touchstart', pf, { passive: true });
  }

  // ---- Pricing ------------------------------------------------------------
  if (page === 'buy') {
    var coupon = getCookie('coupon');
    var cbox = $('#coupon');
    if (coupon && cbox) {
      $('#couponcode').textContent = coupon;
      $('#btncode').href = store.url(store.PRODUCT_REG, 1, coupon);
      cbox.classList.add('show');
    }
    $$('[data-copy]').forEach(function (el) {
      el.addEventListener('click', function () {
        if (navigator.clipboard) navigator.clipboard.writeText(el.textContent.trim());
        var o = el.getAttribute('title'); el.setAttribute('title', 'Copied!');
        setTimeout(function () { el.setAttribute('title', o); }, 1500);
      });
    });
    $$('[data-plan]').forEach(function (plan) {
      var base = +plan.getAttribute('data-base');
      var id = +plan.getAttribute('data-product');
      var tiers = JSON.parse(plan.getAttribute('data-tiers')); // [[maxQty, unitPrice], ...]
      var inp = $('input', plan), btn = $('.buy', plan), unit = $('.unit', plan), rows = $$('.tiers [data-max]', plan);
      function update() {
        var q = Math.round(+inp.value || 1); q = Math.max(1, Math.min(99, q)); inp.value = q;
        var price = base;
        for (var i = 0; i < tiers.length; i++) { if (q <= tiers[i][0]) { price = tiers[i][1]; break; } }
        var total = q * price;
        btn.querySelector('span').textContent = 'Buy Now ($' + total.toLocaleString('en-US') + ')';
        btn.href = store.url(id, q);
        var pct = Math.round((1 - price / base) * 100);
        unit.innerHTML = '<b>$' + price + '</b> / license' + (pct > 0 ? '<br><span class="save">You save ' + pct + '%</span>' : '');
        rows.forEach(function (r) {
          var mx = +r.getAttribute('data-max'), mn = +r.getAttribute('data-min');
          r.classList.toggle('on', q >= mn && q <= mx);
        });
      }
      inp.addEventListener('input', function () { if (inp.value !== '') update(); });
      inp.addEventListener('change', update);
      $$('[data-step]', plan).forEach(function (b) {
        b.addEventListener('click', function () { inp.value = (+inp.value || 1) + +b.getAttribute('data-step'); update(); });
      });
      update();
    });
  }

  // ---- Copy link buttons --------------------------------------------------
  $$('[data-copylink]').forEach(function (b) {
    b.addEventListener('click', function () {
      var lbl = $('span', b), url = b.getAttribute('data-copylink');
      var ok = function () { lbl.textContent = 'Link copied!'; setTimeout(function () { lbl.textContent = 'Copy link'; }, 2000); };
      if (navigator.clipboard) navigator.clipboard.writeText(url).then(ok, function () { prompt('Copy this link:', url); });
      else prompt('Copy this link:', url);
    });
  });

  // ---- Download (auto-start after 2s on Windows, like before) -------------
  var plat = ((navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || '') + ' ' + navigator.userAgent;
  var isWindows = /\bwin(dows|32|64)\b/i.test(plat) && !/windows phone/i.test(plat);
  if (page === 'download' && !isWindows) {
    var dmain = $('.dl:not(.dl-alt)'), alt = $('.dl-alt');
    if (dmain && alt) { dmain.hidden = true; alt.hidden = false; }
    var lead = $('.phero .lead'); if (lead) lead.textContent = 'ScreenToVideo runs on Windows 11, 10, 8 and 7 (64-bit).';
  }
  if (page === 'download' && isWindows) {
    var ring = $('.spinner'), t0 = performance.now(), DUR = 2000;
    (function anim(now) {
      var p = Math.min(1, (now - t0) / DUR);
      if (ring) ring.style.background = 'conic-gradient(var(--coral) ' + (p * 100) + '%, rgba(255,255,255,.08) 0)';
      if (p < 1) requestAnimationFrame(anim);
      else {
        var s = $('#dlstate'); if (s) s.textContent = 'Your download has started';
        location.href = CFG.installer;
      }
    })(t0);
  }

  // ---- Google Drive authorization result ---------------------------------
  if (page === 'gdrive') {
    var st = (qs.get('status') || '').toLowerCase();
    var key = st === 'authorized' ? 'authorized' : st === 'access_denied' ? 'denied' : 'error';
    var el = $('[data-status="' + key + '"]'); if (el) el.classList.add('show');
  }

  // ---- Forms --------------------------------------------------------------
  // ---- Partner code redemption: same license script and parameters as the previous site
  $$('form[data-redeem]').forEach(function (f) {
    f.addEventListener('submit', function (e) {
      e.preventDefault();
      if ($('.hp input', f) && $('.hp input', f).value) return; // bot
      var out = $('.redeem-result', f), msg = $('.form-msg', f), btn = $('button[type=submit]', f);
      var q = ['code', 'email', 'first', 'last'].map(function (k) { return k + '=' + encodeURIComponent(f.elements[k].value.trim()); }).join('&');
      msg.className = 'form-msg'; msg.textContent = '';
      out.hidden = false; out.className = 'redeem-result';
      out.textContent = 'Checking your ' + f.getAttribute('data-partner') + ' code\u2026';
      if (btn) btn.disabled = true;
      fetch(f.getAttribute('data-redeem') + '?' + q)
        .then(function (r) { if (!r.ok) throw 0; return r.text(); })
        .then(function (html) { out.innerHTML = html; }) // HTML from our own license script, shown as on the previous site
        .catch(function () {
          out.hidden = true;
          msg.className = 'form-msg err';
          msg.textContent = 'We couldn\u2019t check your code right now. Please try again, or email ' + CFG.support + ' with your code.';
        })
        .then(function () { if (btn) btn.disabled = false; });
    });
  });

  $$('form[data-form]').forEach(function (f) {
    f.addEventListener('submit', function (e) {
      e.preventDefault();
      var msg = $('.form-msg', f);
      if ($('.hp input', f) && $('.hp input', f).value) return; // bot
      var data = new FormData(f), lines = [];
      data.forEach(function (v, k) { if (k !== '_gotcha' && String(v).trim()) lines.push(k + ': ' + v); });
      var subject = f.getAttribute('data-subject') || 'Website form';
      var subj = data.get('Subject'); if (subj) subject += ' - ' + subj;
      var btn = $('button[type=submit]', f);
      var endpoint = f.getAttribute('data-endpoint') || CFG.formEndpoint;
      function done(ok, text) { msg.className = 'form-msg ' + (ok ? 'ok' : 'err'); msg.textContent = text; if (btn) btn.disabled = false; }
      if (endpoint) {
        if (btn) btn.disabled = true;
        data.append('_subject', subject);
        fetch(endpoint, { method: 'POST', body: data, headers: { Accept: 'application/json' } })
          .then(function (r) { if (!r.ok) throw 0; f.reset(); done(true, f.getAttribute('data-ok') || 'Thank you! Your message has been sent.'); })
          .catch(function () { done(false, 'Something went wrong. Please email us at ' + CFG.support + '.'); });
      } else {
        location.href = 'mailto:' + (f.getAttribute('data-to') || CFG.support) + '?subject=' + encodeURIComponent(subject) +
          '&body=' + encodeURIComponent(lines.join('\n'));
        done(true, 'Your email app should open with the message ready to send. If it doesn’t, write to us at ' + (f.getAttribute('data-to') || CFG.support) + '.');
      }
    });
  });

  // ---- Consent + analytics + live chat (deferred until idle) --------------
  window.dataLayer = window.dataLayer || [];
  function gtag() { dataLayer.push(arguments); }
  window.gtag = gtag;
  var consent = ls('s2v-consent');
  gtag('consent', 'default', {
    ad_storage: consent === 'yes' ? 'granted' : 'denied',
    ad_user_data: consent === 'yes' ? 'granted' : 'denied',
    ad_personalization: consent === 'yes' ? 'granted' : 'denied',
    analytics_storage: consent === 'yes' ? 'granted' : 'denied'
  });
  gtag('js', new Date());
  gtag('config', CFG.adsId);
  if (page === 'download') gtag('event', 'conversion', { send_to: CFG.downloadConversion, value: 1.0, currency: 'RON' });

  var cb = $('.consent');
  if (cb && !consent) {
    setTimeout(function () { cb.classList.add('show'); }, 1200);
    $$('[data-consent]', cb).forEach(function (b) {
      b.addEventListener('click', function () {
        var yes = b.getAttribute('data-consent') === 'yes';
        ls('s2v-consent', yes ? 'yes' : 'no');
        if (yes) gtag('consent', 'update', { ad_storage: 'granted', ad_user_data: 'granted', ad_personalization: 'granted', analytics_storage: 'granted' });
        cb.classList.remove('show');
      });
    });
  }

  function addScript(src, attrs) {
    var s = d.createElement('script'); s.src = src; s.async = true;
    for (var k in attrs || {}) s.setAttribute(k, attrs[k]);
    d.head.appendChild(s);
  }
  idle(function () {
    addScript('https://www.googletagmanager.com/gtag/js?id=' + CFG.adsId);
    if (!body.hasAttribute('data-nochat')) { window.Tawk_API = window.Tawk_API || {}; window.Tawk_LoadStart = new Date(); addScript(CFG.tawk, { charset: 'UTF-8', crossorigin: '*' }); }
  }, page === 'download' ? 0 : 2500);
})();
