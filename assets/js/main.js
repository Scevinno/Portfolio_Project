/* Nikolas Scevko — portfolio interactions (vanilla JS, no dependencies) */
(function () {
  "use strict";

  var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- rotating role in the hero (type / erase cycle) ---- */
  var rotator = document.getElementById("role-rotator");
  if (rotator && !reducedMotion) {
    var roles;
    try { roles = JSON.parse(rotator.getAttribute("data-roles")) || []; }
    catch (e) { roles = []; }

    if (roles.length > 1) {
      var roleIdx = 0, charIdx = roles[0].length, deleting = false;

      var tick = function () {
        var current = roles[roleIdx];
        if (deleting) {
          charIdx--;
          if (charIdx <= 0) {
            deleting = false;
            roleIdx = (roleIdx + 1) % roles.length;
            current = roles[roleIdx];
          }
        } else {
          charIdx++;
          if (charIdx >= current.length) {
            charIdx = current.length;
            deleting = true;
            rotator.textContent = current;
            setTimeout(tick, 2200); // pause on the full word
            return;
          }
        }
        rotator.textContent = current.slice(0, charIdx);
        setTimeout(tick, deleting ? 45 : 85);
      };

      setTimeout(tick, 2200);
    }
  }

  /* ---- scroll-reveal ---- */
  var revealEls = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window && !reducedMotion) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    revealEls.forEach(function (el) { io.observe(el); });
  } else {
    revealEls.forEach(function (el) { el.classList.add("is-visible"); });
  }

  /* ---- reading progress bar (post pages) ---- */
  var bar = document.getElementById("progress-bar");
  if (bar) {
    var onScroll = function () {
      var doc = document.documentElement;
      var total = doc.scrollHeight - doc.clientHeight;
      var pct = total > 0 ? (doc.scrollTop / total) * 100 : 0;
      bar.style.width = pct + "%";
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }
})();
