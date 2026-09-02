"use strict";


let selectedSpecies = null;


const emergencyGrid = document.getElementById(
    "emergency-grid"
);

const speciesButtons = document.querySelectorAll(
    ".species-button"
);

const speciesStatus = document.getElementById(
    "species-status"
);

const askForm = document.getElementById(
    "ask-form"
);

const questionInput = document.getElementById(
    "question-input"
);

const submitButton = document.getElementById(
    "submit-button"
);

const loadingPanel = document.getElementById(
    "loading-panel"
);

const errorPanel = document.getElementById(
    "error-panel"
);

const errorMessage = document.getElementById(
    "error-message"
);

const answerPanel = document.getElementById(
    "answer-panel"
);

const answerText = document.getElementById(
    "answer-text"
);

const sourcesList = document.getElementById(
    "sources-list"
);


function setSpecies(species) {
    selectedSpecies = species || null;

    speciesButtons.forEach((button) => {
        const buttonSpecies =
            button.dataset.species || null;

        const isSelected =
            buttonSpecies === selectedSpecies;

        button.classList.toggle(
            "active",
            isSelected
        );

        button.setAttribute(
            "aria-pressed",
            String(isSelected)
        );
    });

    if (selectedSpecies === "dog") {
        speciesStatus.textContent =
            "Species: dog";
    } else if (selectedSpecies === "cat") {
        speciesStatus.textContent =
            "Species: cat";
    } else {
        speciesStatus.textContent =
            "Species: not specified";
    }
}


speciesButtons.forEach((button) => {
    button.addEventListener(
        "click",
        () => {
            setSpecies(
                button.dataset.species
            );
        }
    );
});


function createEmergencyCard(condition) {
    const button = document.createElement(
        "button"
    );

    button.type = "button";
    button.className = "emergency-card";

    const urgency = document.createElement(
        "span"
    );

    urgency.className =
        `urgency-badge ${condition.urgency}`;

    urgency.textContent =
        condition.urgency;

    const title = document.createElement(
        "span"
    );

    title.className =
        "emergency-card-title";

    title.textContent =
        condition.title;

    const description =
        document.createElement(
            "span"
        );

    description.className =
        "emergency-card-description";

    description.textContent =
        condition.short_description;

    button.append(
        urgency,
        title,
        description
    );

    button.addEventListener(
        "click",
        () => {
            questionInput.value =
                condition.starter_question;

            questionInput.focus();

            questionInput.scrollIntoView({
                behavior: "smooth",
                block: "center"
            });
        }
    );

    return button;
}


async function loadEmergencyConditions() {
    try {
        const response = await fetch(
            "/emergencies"
        );

        if (!response.ok) {
            throw new Error(
                "Emergency options could not be loaded."
            );
        }

        const payload = await response.json();

        emergencyGrid.replaceChildren();

        payload.conditions.forEach(
            (condition) => {
                emergencyGrid.appendChild(
                    createEmergencyCard(
                        condition
                    )
                );
            }
        );

    } catch (error) {
        emergencyGrid.replaceChildren();

        const message = document.createElement(
            "p"
        );

        message.className = "muted";

        message.textContent =
            "Emergency shortcuts are temporarily unavailable. " +
            "You can still describe what is happening below.";

        emergencyGrid.appendChild(
            message
        );
    }
}


function cleanAnswerFormatting(text) {
    return text
        .replace(
            /^#{1,6}\s+/gm,
            ""
        )
        .replace(
            /\*\*(.*?)\*\*/g,
            "$1"
        )
        .replace(
            /`([^`]+)`/g,
            "$1"
        )
        .trim();
}


function safeHttpUrl(value) {
    if (!value) {
        return null;
    }

    try {
        const url = new URL(
            value,
            window.location.origin
        );

        if (
            url.protocol !== "http:"
            &&
            url.protocol !== "https:"
        ) {
            return null;
        }

        return url.href;

    } catch {
        return null;
    }
}


function renderSources(sources) {
    sourcesList.replaceChildren();

    if (!sources || sources.length === 0) {
        const empty = document.createElement(
            "p"
        );

        empty.className = "muted";

        empty.textContent =
            "No source metadata was returned.";

        sourcesList.appendChild(
            empty
        );

        return;
    }

    sources.forEach((source) => {
        const card = document.createElement(
            "article"
        );

        card.className = "source-card";

        const label = document.createElement(
            "div"
        );

        label.className = "source-label";

        label.textContent =
            `[${source.label}]`;

        const title = document.createElement(
            "div"
        );

        title.className = "source-title";

        title.textContent =
            source.title || "Veterinary source";

        const publisher =
            document.createElement(
                "p"
            );

        publisher.className =
            "source-meta";

        publisher.textContent =
            source.publisher || "";

        card.append(
            label,
            title,
            publisher
        );

        if (source.section) {
            const section =
                document.createElement(
                    "p"
                );

            section.className =
                "source-meta";

            section.textContent =
                `Section: ${source.section}`;

            card.appendChild(
                section
            );
        }

        const validUrl = safeHttpUrl(
            source.url
        );

        if (validUrl) {
            const link =
                document.createElement(
                    "a"
                );

            link.className =
                "source-link";

            link.href = validUrl;

            link.target = "_blank";

            link.rel =
                "noopener noreferrer";

            link.textContent =
                "Open source";

            card.appendChild(
                link
            );
        }

        sourcesList.appendChild(
            card
        );
    });
}


function showLoading() {
    errorPanel.classList.add(
        "hidden"
    );

    answerPanel.classList.add(
        "hidden"
    );

    loadingPanel.classList.remove(
        "hidden"
    );

    submitButton.disabled = true;

    submitButton.textContent =
        "Getting guidance...";
}


function stopLoading() {
    loadingPanel.classList.add(
        "hidden"
    );

    submitButton.disabled = false;

    submitButton.textContent =
        "Get first-aid guidance";
}


function showError(message) {
    answerPanel.classList.add(
        "hidden"
    );

    errorMessage.textContent =
        message;

    errorPanel.classList.remove(
        "hidden"
    );

    errorPanel.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


function showAnswer(payload) {
    const cleanedAnswer =
        cleanAnswerFormatting(
            payload.answer || ""
        );

    answerText.textContent =
        cleanedAnswer;

    renderSources(
        payload.sources || []
    );

    errorPanel.classList.add(
        "hidden"
    );

    answerPanel.classList.remove(
        "hidden"
    );

    answerPanel.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}


async function askAssistant(question) {
    showLoading();

    try {
        const response = await fetch(
            "/ask",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    question: question,
                    species: selectedSpecies
                })
            }
        );

        const payload =
            await response.json();

        if (!response.ok) {
            const detail =
                typeof payload.detail ===
                "string"
                    ? payload.detail
                    : (
                        "The assistant is temporarily unavailable. " +
                        "Contact a veterinarian directly if this is urgent."
                    );

            throw new Error(
                detail
            );
        }

        showAnswer(
            payload
        );

    } catch (error) {
        showError(
            error.message ||
            (
                "The assistant is temporarily unavailable. " +
                "Contact a veterinarian directly if this is urgent."
            )
        );

    } finally {
        stopLoading();
    }
}


askForm.addEventListener(
    "submit",
    async (event) => {
        event.preventDefault();

        const question =
            questionInput.value.trim();

        if (question.length < 3) {
            showError(
                "Please describe what is happening with your pet."
            );

            return;
        }

        await askAssistant(
            question
        );
    }
);


setSpecies(
    null
);

loadEmergencyConditions();