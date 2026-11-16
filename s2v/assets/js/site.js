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
      list = $$('.ftile[data-video], .tut[data-video]').filter(function (x) { return x.offsetParent !== null; });
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
  if (sn && 'IntersectionObserver' in window) {
    var links = $$('a', sn), map = {};
    links.forEach(function (a) { map[a.getAttribute('href').slice(1)] = a; });
    var spy = new IntersectionObserver(function (es) {
      es.forEach(function (e) {
        if (e.isIntersecting) {
          links.forEach(function (a) { a.classList.remove('on'); });
          var a = map[e.target.id]; if (a) { a.classList.add('on'); a.scrollIntoView({ block: 'nearest', inline: 'nearest' }); }
        }
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    Object.keys(map).forEach(function (id) { var s = d.getElementById(id); if (s) spy.observe(s); });
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

  // ---- Download (auto-start after 2s, like before) -----------------------
  if (page === 'download') {
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
