document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".accordion-title").forEach(title => {
    title.addEventListener("click", () => {
      const parent = title.closest(".accordion");
      parent.classList.toggle("open");
    });
  });
});

