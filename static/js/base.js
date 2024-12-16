document.addEventListener("DOMContentLoaded", function () {
    // Select all elements to be lazy-loaded or animated
    const elementsToObserve = document.querySelectorAll(".fade-in, .lazy-loading");

    const observer = new IntersectionObserver(
        (entries, observer) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("visible");
                    // Stop observing once the element is visible
                    observer.unobserve(entry.target);
                }
            });
        },
        {
            // Trigger when 20% of the element is visible
            threshold: 0.2,
        }
    );

    // Observe all selected elements
    elementsToObserve.forEach((element) => {
        observer.observe(element);
    });
});