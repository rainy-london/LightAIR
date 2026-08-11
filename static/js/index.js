(() => {
  const header = document.querySelector("[data-header]");
  const menuToggle = document.querySelector("[data-menu-toggle]");
  const nav = document.querySelector("[data-nav]");
  const menuPath = menuToggle?.querySelector("path");

  const setMenuState = (open) => {
    if (!menuToggle || !nav || !menuPath) return;
    nav.classList.toggle("is-open", open);
    document.body.classList.toggle("menu-open", open);
    menuToggle.setAttribute("aria-expanded", String(open));
    menuToggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");
    menuPath.setAttribute("d", open ? "M18 6 6 18M6 6l12 12" : "M4 7h16M4 12h16M4 17h16");
  };

  menuToggle?.addEventListener("click", () => {
    setMenuState(menuToggle.getAttribute("aria-expanded") !== "true");
  });

  nav?.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => setMenuState(false));
  });

  window.addEventListener("resize", () => {
    if (window.innerWidth > 1020) setMenuState(false);
  });

  const updateHeader = () => {
    header?.classList.toggle("is-scrolled", window.scrollY > 12);
  };

  updateHeader();
  window.addEventListener("scroll", updateHeader, { passive: true });

  const copyButton = document.querySelector("[data-copy-bibtex]");
  const copyStatus = document.querySelector("[data-copy-status]");
  const bibtex = document.querySelector("#bibtex-code");

  const fallbackCopy = (text) => {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();
    return copied;
  };

  copyButton?.addEventListener("click", async () => {
    const text = bibtex?.textContent?.trim();
    if (!text || !copyStatus) return;

    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
      } else if (!fallbackCopy(text)) {
        throw new Error("Clipboard unavailable");
      }
      copyStatus.textContent = "BibTeX copied.";
    } catch {
      copyStatus.textContent = "Copy failed. Select the citation manually.";
    }

    window.setTimeout(() => {
      copyStatus.textContent = "";
    }, 2600);
  });

  const dialog = document.querySelector("[data-figure-dialog]");
  const dialogImage = document.querySelector("[data-dialog-image]");
  const dialogCaption = document.querySelector("[data-dialog-caption]");
  const dialogClose = document.querySelector("[data-dialog-close]");

  document.querySelectorAll("[data-zoom-src]").forEach((button) => {
    button.addEventListener("click", () => {
      if (!(dialog instanceof HTMLDialogElement) || !(dialogImage instanceof HTMLImageElement)) return;
      dialogImage.src = button.dataset.zoomSrc || "";
      dialogImage.alt = button.dataset.zoomAlt || "Expanded paper figure";
      if (dialogCaption) dialogCaption.textContent = button.dataset.zoomCaption || "";
      dialog.showModal();
    });
  });

  dialogClose?.addEventListener("click", () => dialog?.close());

  dialog?.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });

  dialog?.addEventListener("close", () => {
    if (dialogImage instanceof HTMLImageElement) {
      dialogImage.src = "";
      dialogImage.alt = "";
    }
  });
})();
