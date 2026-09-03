"use strict";


let latestDashboardData = null;


function getCanvasContext(canvasId) {
    const canvas = document.getElementById(canvasId);

    const ratio = window.devicePixelRatio || 1;

    const width = canvas.clientWidth || 600;
    const height = canvas.clientHeight || 280;

    canvas.width = width * ratio;
    canvas.height = height * ratio;

    const context = canvas.getContext("2d");

    context.scale(ratio, ratio);

    return {
        canvas,
        context,
        width,
        height,
    };
}


function clearCanvas(context, width, height) {
    context.clearRect(
        0,
        0,
        width,
        height,
    );
}


function getCssColor(variableName, fallback) {
    const value = getComputedStyle(
        document.documentElement
    ).getPropertyValue(
        variableName
    ).trim();

    return value || fallback;
}


function chartColors() {
    return {
        primary: getCssColor(
            "--dashboard-primary",
            "#2f6f5e",
        ),
        secondary: getCssColor(
            "--dashboard-secondary",
            "#7ca89c",
        ),
        text: getCssColor(
            "--dashboard-text",
            "#25332f",
        ),
        muted: getCssColor(
            "--dashboard-muted",
            "#70817b",
        ),
        grid: getCssColor(
            "--dashboard-grid",
            "#dbe5e1",
        ),
        positive: getCssColor(
            "--dashboard-positive",
            "#438a68",
        ),
        negative: getCssColor(
            "--dashboard-negative",
            "#b86161",
        ),
    };
}


function formatShortDate(value) {
    const date = new Date(
        `${value}T00:00:00`
    );

    if (Number.isNaN(date.getTime())) {
        return value;
    }

    return date.toLocaleDateString(
        undefined,
        {
            month: "short",
            day: "numeric",
        }
    );
}


function drawEmptyState(
    context,
    width,
    height,
    message,
) {
    const colors = chartColors();

    clearCanvas(
        context,
        width,
        height,
    );

    context.fillStyle = colors.muted;
    context.font = "14px sans-serif";
    context.textAlign = "center";
    context.textBaseline = "middle";

    context.fillText(
        message,
        width / 2,
        height / 2,
    );
}


function drawAxes(
    context,
    width,
    height,
    padding,
) {
    const colors = chartColors();

    context.strokeStyle = colors.grid;
    context.lineWidth = 1;

    context.beginPath();

    context.moveTo(
        padding.left,
        padding.top,
    );

    context.lineTo(
        padding.left,
        height - padding.bottom,
    );

    context.lineTo(
        width - padding.right,
        height - padding.bottom,
    );

    context.stroke();
}


function drawLineChart(
    canvasId,
    points,
    options = {},
) {
    const {
        context,
        width,
        height,
    } = getCanvasContext(
        canvasId
    );

    const colors = chartColors();

    const validPoints = points.filter(
        (point) => (
            point.value !== null
            && point.value !== undefined
        )
    );

    if (validPoints.length === 0) {
        drawEmptyState(
            context,
            width,
            height,
            "No data available yet.",
        );

        return;
    }

    clearCanvas(
        context,
        width,
        height,
    );

    const padding = {
        top: 24,
        right: 20,
        bottom: 45,
        left: 52,
    };

    drawAxes(
        context,
        width,
        height,
        padding,
    );

    const values = validPoints.map(
        (point) => Number(point.value)
    );

    const maximum = Math.max(
        ...values,
        1,
    );

    const chartWidth = (
        width
        - padding.left
        - padding.right
    );

    const chartHeight = (
        height
        - padding.top
        - padding.bottom
    );

    const denominator = Math.max(
        validPoints.length - 1,
        1,
    );

    const coordinates = validPoints.map(
        (point, index) => {
            const x = (
                padding.left
                + (
                    index
                    / denominator
                )
                * chartWidth
            );

            const y = (
                padding.top
                + chartHeight
                - (
                    Number(point.value)
                    / maximum
                )
                * chartHeight
            );

            return {
                x,
                y,
                point,
            };
        }
    );

    context.strokeStyle = colors.primary;
    context.lineWidth = 3;
    context.lineJoin = "round";
    context.lineCap = "round";

    context.beginPath();

    coordinates.forEach(
        (coordinate, index) => {
            if (index === 0) {
                context.moveTo(
                    coordinate.x,
                    coordinate.y,
                );
            } else {
                context.lineTo(
                    coordinate.x,
                    coordinate.y,
                );
            }
        }
    );

    context.stroke();

    coordinates.forEach(
        (coordinate) => {
            context.fillStyle = colors.primary;

            context.beginPath();

            context.arc(
                coordinate.x,
                coordinate.y,
                4,
                0,
                Math.PI * 2,
            );

            context.fill();
        }
    );

    context.fillStyle = colors.muted;
    context.font = "12px sans-serif";
    context.textAlign = "center";

    coordinates.forEach(
        (coordinate) => {
            context.fillText(
                formatShortDate(
                    coordinate.point.label
                ),
                coordinate.x,
                height - 18,
            );
        }
    );

    context.textAlign = "right";

    context.fillText(
        options.valueFormatter
            ? options.valueFormatter(maximum)
            : Math.round(maximum).toString(),
        padding.left - 8,
        padding.top + 4,
    );

    context.fillText(
        "0",
        padding.left - 8,
        height - padding.bottom + 4,
    );
}


function drawVerticalBarChart(
    canvasId,
    points,
) {
    const {
        context,
        width,
        height,
    } = getCanvasContext(
        canvasId
    );

    const colors = chartColors();

    if (
        !points.length
        || points.every(
            (point) => Number(point.value) === 0
        )
    ) {
        drawEmptyState(
            context,
            width,
            height,
            "No data available yet.",
        );

        return;
    }

    clearCanvas(
        context,
        width,
        height,
    );

    const padding = {
        top: 24,
        right: 20,
        bottom: 52,
        left: 42,
    };

    drawAxes(
        context,
        width,
        height,
        padding,
    );

    const maximum = Math.max(
        ...points.map(
            (point) => Number(point.value)
        ),
        1,
    );

    const chartWidth = (
        width
        - padding.left
        - padding.right
    );

    const chartHeight = (
        height
        - padding.top
        - padding.bottom
    );

    const slotWidth = (
        chartWidth
        / Math.max(
            points.length,
            1,
        )
    );

    const barWidth = Math.min(
        64,
        slotWidth * 0.58,
    );

    points.forEach(
        (point, index) => {
            const value = Number(
                point.value
            );

            const heightRatio = (
                value / maximum
            );

            const barHeight = (
                chartHeight
                * heightRatio
            );

            const x = (
                padding.left
                + index * slotWidth
                + (
                    slotWidth
                    - barWidth
                ) / 2
            );

            const y = (
                padding.top
                + chartHeight
                - barHeight
            );

            context.fillStyle = colors.primary;

            context.fillRect(
                x,
                y,
                barWidth,
                barHeight,
            );

            context.fillStyle = colors.text;
            context.font = "13px sans-serif";
            context.textAlign = "center";

            context.fillText(
                value.toString(),
                x + barWidth / 2,
                Math.max(
                    y - 8,
                    14,
                ),
            );

            context.fillStyle = colors.muted;
            context.font = "12px sans-serif";

            context.fillText(
                point.label,
                x + barWidth / 2,
                height - 20,
            );
        }
    );
}


function drawFeedbackChart(
    canvasId,
    points,
) {
    const {
        context,
        width,
        height,
    } = getCanvasContext(
        canvasId
    );

    const colors = chartColors();

    const total = points.reduce(
        (
            sum,
            point,
        ) => (
            sum
            + Number(point.value)
        ),
        0,
    );

    if (total === 0) {
        drawEmptyState(
            context,
            width,
            height,
            "No feedback has been submitted yet.",
        );

        return;
    }

    clearCanvas(
        context,
        width,
        height,
    );

    const centerX = width / 2;
    const centerY = height / 2 - 8;

    const radius = Math.min(
        width,
        height,
    ) * 0.28;

    let startAngle = -Math.PI / 2;

    points.forEach(
        (point) => {
            const value = Number(
                point.value
            );

            const angle = (
                value
                / total
            ) * Math.PI * 2;

            const isPositive = (
                point.label.toLowerCase()
                === "positive"
            );

            context.fillStyle = (
                isPositive
                    ? colors.positive
                    : colors.negative
            );

            context.beginPath();

            context.moveTo(
                centerX,
                centerY,
            );

            context.arc(
                centerX,
                centerY,
                radius,
                startAngle,
                startAngle + angle,
            );

            context.closePath();
            context.fill();

            startAngle += angle;
        }
    );

    context.fillStyle = (
        getCssColor(
            "--dashboard-card",
            "#ffffff",
        )
    );

    context.beginPath();

    context.arc(
        centerX,
        centerY,
        radius * 0.58,
        0,
        Math.PI * 2,
    );

    context.fill();

    context.fillStyle = colors.text;
    context.textAlign = "center";
    context.font = "bold 24px sans-serif";

    context.fillText(
        total.toString(),
        centerX,
        centerY,
    );

    context.fillStyle = colors.muted;
    context.font = "12px sans-serif";

    context.fillText(
        "feedback",
        centerX,
        centerY + 20,
    );

    const legendY = height - 20;

    points.forEach(
        (point, index) => {
            const isPositive = (
                point.label.toLowerCase()
                === "positive"
            );

            context.fillStyle = (
                isPositive
                    ? colors.positive
                    : colors.negative
            );

            const x = (
                width / 2
                - 95
                + index * 120
            );

            context.fillRect(
                x,
                legendY - 9,
                10,
                10,
            );

            context.fillStyle = colors.text;
            context.textAlign = "left";
            context.font = "12px sans-serif";

            context.fillText(
                `${point.label}: ${point.value}`,
                x + 16,
                legendY,
            );
        }
    );
}


function truncateLabel(
    label,
    maximumLength = 34,
) {
    if (
        label.length
        <= maximumLength
    ) {
        return label;
    }

    return (
        label.slice(
            0,
            maximumLength - 1
        )
        + "…"
    );
}


function drawHorizontalBarChart(
    canvasId,
    points,
) {
    const {
        context,
        width,
        height,
    } = getCanvasContext(
        canvasId
    );

    const colors = chartColors();

    if (
        !points.length
        || points.every(
            (point) => Number(point.value) === 0
        )
    ) {
        drawEmptyState(
            context,
            width,
            height,
            "No retrieved-source data available yet.",
        );

        return;
    }

    clearCanvas(
        context,
        width,
        height,
    );

    const padding = {
        top: 20,
        right: 48,
        bottom: 20,
        left: Math.min(
            220,
            width * 0.38,
        ),
    };

    const chartWidth = (
        width
        - padding.left
        - padding.right
    );

    const availableHeight = (
        height
        - padding.top
        - padding.bottom
    );

    const maximum = Math.max(
        ...points.map(
            (point) => Number(point.value)
        ),
        1,
    );

    const rowHeight = (
        availableHeight
        / Math.max(
            points.length,
            1,
        )
    );

    points.forEach(
        (point, index) => {
            const value = Number(
                point.value
            );

            const barWidth = (
                chartWidth
                * (
                    value
                    / maximum
                )
            );

            const barHeight = Math.min(
                34,
                rowHeight * 0.56,
            );

            const y = (
                padding.top
                + index * rowHeight
                + (
                    rowHeight
                    - barHeight
                ) / 2
            );

            context.fillStyle = colors.text;
            context.font = "12px sans-serif";
            context.textAlign = "right";
            context.textBaseline = "middle";

            context.fillText(
                truncateLabel(
                    point.label
                ),
                padding.left - 12,
                y + barHeight / 2,
            );

            context.fillStyle = colors.primary;

            context.fillRect(
                padding.left,
                y,
                barWidth,
                barHeight,
            );

            context.fillStyle = colors.text;
            context.textAlign = "left";

            context.fillText(
                value.toString(),
                padding.left
                    + barWidth
                    + 8,
                y + barHeight / 2,
            );
        }
    );
}


function buildSummary(points) {
    if (!points.length) {
        return "No observations yet.";
    }

    return points
        .map(
            (point) => (
                `${point.label}: ${
                    point.value ?? "no data"
                }`
            )
        )
        .join(" · ");
}


function renderMetrics(metrics) {
    document.getElementById(
        "metric-total-requests"
    ).textContent = (
        metrics.total_requests
    );

    document.getElementById(
        "metric-last-24h"
    ).textContent = (
        metrics.requests_last_24_hours
    );

    document.getElementById(
        "metric-latency"
    ).textContent = (
        metrics.average_latency_ms === null
            ? "—"
            : `${
                Math.round(
                    metrics.average_latency_ms
                )
            } ms`
    );

    document.getElementById(
        "metric-feedback-total"
    ).textContent = (
        metrics.feedback_total
    );

    document.getElementById(
        "metric-positive-rate"
    ).textContent = (
        metrics.positive_feedback_rate === null
            ? "—"
            : `${
                Math.round(
                    metrics.positive_feedback_rate
                    * 100
                )
            }%`
    );
}


function renderDashboard(data) {
    latestDashboardData = data;

    drawLineChart(
        "requests-chart",
        data.requests_by_day,
    );

    drawLineChart(
        "latency-chart",
        data.latency_by_day,
        {
            valueFormatter: (
                value
            ) => (
                `${Math.round(value)} ms`
            ),
        },
    );

    drawVerticalBarChart(
        "species-chart",
        data.species_breakdown,
    );

    drawFeedbackChart(
        "feedback-chart",
        data.feedback_breakdown,
    );

    drawHorizontalBarChart(
        "sources-chart",
        data.top_sources,
    );

    document.getElementById(
        "requests-chart-summary"
    ).textContent = buildSummary(
        data.requests_by_day
    );

    document.getElementById(
        "latency-chart-summary"
    ).textContent = buildSummary(
        data.latency_by_day
    );

    document.getElementById(
        "species-chart-summary"
    ).textContent = buildSummary(
        data.species_breakdown
    );

    document.getElementById(
        "feedback-chart-summary"
    ).textContent = buildSummary(
        data.feedback_breakdown
    );

    document.getElementById(
        "sources-chart-summary"
    ).textContent = buildSummary(
        data.top_sources
    );
}


function showError(message) {
    const element = document.getElementById(
        "dashboard-error"
    );

    element.textContent = message;

    element.classList.remove(
        "hidden"
    );
}


async function loadDashboard() {
    try {
        const [
            metricsResponse,
            dashboardResponse,
        ] = await Promise.all(
            [
                fetch("/metrics"),
                fetch("/dashboard-data"),
            ]
        );

        if (
            !metricsResponse.ok
            || !dashboardResponse.ok
        ) {
            throw new Error(
                "Monitoring data could not be loaded."
            );
        }

        const metrics = (
            await metricsResponse.json()
        );

        const dashboardData = (
            await dashboardResponse.json()
        );

        renderMetrics(
            metrics
        );

        if (
            !metrics.enabled
            || !dashboardData.enabled
        ) {
            document.getElementById(
                "monitoring-disabled"
            ).classList.remove(
                "hidden"
            );
        }

        renderDashboard(
            dashboardData
        );

    } catch (error) {
        console.error(
            error
        );

        showError(
            "The monitoring dashboard is temporarily unavailable."
        );
    }
}


let resizeTimer = null;

window.addEventListener(
    "resize",
    () => {
        window.clearTimeout(
            resizeTimer
        );

        resizeTimer = window.setTimeout(
            () => {
                if (
                    latestDashboardData
                ) {
                    renderDashboard(
                        latestDashboardData
                    );
                }
            },
            150,
        );
    }
);


document.addEventListener(
    "DOMContentLoaded",
    loadDashboard,
);