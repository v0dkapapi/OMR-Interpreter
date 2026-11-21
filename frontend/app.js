/**
 * OMR Interpreter Frontend Application Logic
 * Pure Vanilla JavaScript
 * Zero Emojis, Strict White & Red Design System
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Elements ---
  const dropzone = document.getElementById("dropzone");
  const imageInput = document.getElementById("imageInput");
  const fileSelectedDisplay = document.getElementById("fileSelectedDisplay");
  const selectedFileName = document.getElementById("selectedFileName");
  const clearFileBtn = document.getElementById("clearFileBtn");

  const answerKeySelect = document.getElementById("answerKeySelect");
  const customKeyUploadGroup = document.getElementById("customKeyUploadGroup");
  const keyFileInput = document.getElementById("keyFileInput");
  const marksCorrectInput = document.getElementById("marksCorrectInput");
  const negativeMarksInput = document.getElementById("negativeMarksInput");
  const multiMarkInput = document.getElementById("multiMarkInput");
  const processBtn = document.getElementById("processBtn");

  const loadingState = document.getElementById("loadingState");
  const loadingStageTitle = document.getElementById("loadingStageTitle");
  const loadingStageDesc = document.getElementById("loadingStageDesc");

  const errorState = document.getElementById("errorState");
  const errorMessageTitle = document.getElementById("errorMessageTitle");
  const errorMessageDesc = document.getElementById("errorMessageDesc");

  const resultsSection = document.getElementById("resultsSection");
  const systemStatusText = document.getElementById("systemStatusText");

  // Summary Metrics
  const marksObtainedVal = document.getElementById("marksObtainedVal");
  const totalMarksVal = document.getElementById("totalMarksVal");
  const percentageVal = document.getElementById("percentageVal");
  const percentageBar = document.getElementById("percentageBar");
  const correctCountVal = document.getElementById("correctCountVal");
  const incorrectCountVal = document.getElementById("incorrectCountVal");
  const multiMarkCountVal = document.getElementById("multiMarkCountVal");
  const unansweredCountVal = document.getElementById("unansweredCountVal");

  // Student Metadata
  const ocrEngineBadge = document.getElementById("ocrEngineBadge");
  const studentIdVal = document.getElementById("studentIdVal");
  const studentIdConf = document.getElementById("studentIdConf");
  const studentNameVal = document.getElementById("studentNameVal");
  const studentNameConf = document.getElementById("studentNameConf");
  const moduleCodeVal = document.getElementById("moduleCodeVal");
  const moduleCodeConf = document.getElementById("moduleCodeConf");
  const seatNumberVal = document.getElementById("seatNumberVal");
  const seatNumberConf = document.getElementById("seatNumberConf");
  const examDateVal = document.getElementById("examDateVal");
  const examDateConf = document.getElementById("examDateConf");
  const alignmentConfidenceVal = document.getElementById("alignmentConfidenceVal");
  const alignmentStatusBadge = document.getElementById("alignmentStatusBadge");

  // Issues & Table
  const flaggedCountBadge = document.getElementById("flaggedCountBadge");
  const flaggedIssuesList = document.getElementById("flaggedIssuesList");
  const questionTableBody = document.getElementById("questionTableBody");

  // Preview & Zoom Modal
  const overlayPreviewImage = document.getElementById("overlayPreviewImage");
  const togglePreviewZoomBtn = document.getElementById("togglePreviewZoomBtn");
  const previewModal = document.getElementById("previewModal");
  const modalBackdrop = document.getElementById("modalBackdrop");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const modalPreviewImage = document.getElementById("modalPreviewImage");

  // Export Buttons
  const exportCsvBtn = document.getElementById("exportCsvBtn");
  const exportJsonBtn = document.getElementById("exportJsonBtn");

  // State
  let currentFile = null;
  let currentResultId = null;

  // --- Initialize Health Check ---
  fetch("/api/health")
    .then((res) => res.json())
    .then((data) => {
      if (data.status === "ok") {
        systemStatusText.textContent = `System Ready [${data.device.toUpperCase()}]`;
      }
    })
    .catch(() => {
      systemStatusText.textContent = "Service Disconnected";
    });

  // --- File Selection & Drag/Drop Handling ---
  function handleFileSelected(file) {
    if (!file) return;

    // Validate type
    const validExtensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some((ext) => fileName.endsWith(ext));

    if (!isValid) {
      showError("Invalid File Format", "Please select a supported image file (.jpg, .png, .webp, .bmp, .tiff).");
      return;
    }

    // Validate size (25MB)
    if (file.size > 25 * 1024 * 1024) {
      showError("File Too Large", "Maximum allowed file size is 25 MB.");
      return;
    }

    hideError();
    currentFile = file;
    selectedFileName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    fileSelectedDisplay.classList.remove("hidden");
    dropzone.querySelector(".dropzone-content").classList.add("hidden");
    processBtn.disabled = false;

    // Automatically trigger grading when an image is uploaded or dropped
    triggerGrading();
  }

  function clearSelectedFile() {
    currentFile = null;
    imageInput.value = "";
    selectedFileName.textContent = "";
    fileSelectedDisplay.classList.add("hidden");
    const dropContent = dropzone.querySelector(".dropzone-content");
    if (dropContent) dropContent.classList.remove("hidden");
    processBtn.disabled = true;
    resultsSection.classList.add("hidden");
    hideError();
  }

  clearFileBtn.addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    clearSelectedFile();
  });

  // Click on dropzone to open native file dialog
  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("#clearFileBtn")) {
      return;
    }
    imageInput.value = "";
    imageInput.click();
  });

  // Keyboard accessibility for dropzone (Enter or Space)
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      imageInput.value = "";
      imageInput.click();
    }
  });

  imageInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  // Drag and Drop events
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      handleFileSelected(dt.files[0]);
    }
  });

  // --- Answer Key Selector Toggle ---
  answerKeySelect.addEventListener("change", () => {
    if (answerKeySelect.value === "custom") {
      customKeyUploadGroup.classList.remove("hidden");
    } else {
      customKeyUploadGroup.classList.add("hidden");
      keyFileInput.value = "";
    }
  });

  // --- Error & Loading Helpers ---
  function showError(title, message) {
    errorMessageTitle.textContent = title;
    errorMessageDesc.textContent = message;
    errorState.classList.remove("hidden");
  }

  function hideError() {
    errorState.classList.add("hidden");
  }

  function setLoading(isLoading, title = "Processing Sheet...", desc = "Running perspective alignment and grading.") {
    if (isLoading) {
      loadingStageTitle.textContent = title;
      loadingStageDesc.textContent = desc;
      loadingState.classList.remove("hidden");
      processBtn.disabled = true;
    } else {
      loadingState.classList.add("hidden");
      processBtn.disabled = !currentFile;
    }
  }

  // --- Grading Action ---
  async function triggerGrading() {
    if (!currentFile) return;

    hideError();
    resultsSection.classList.add("hidden");
    setLoading(true, "Aligning Sheet & Normalizing Shadows", "Running SIFT homography and illumination normalization...");

    const formData = new FormData();
    formData.append("image", currentFile);
    formData.append("marks_per_correct", marksCorrectInput.value || "1.0");
    formData.append("negative_marks", negativeMarksInput.value || "0.0");
    formData.append("multi_mark_marks", multiMarkInput.value || "0.0");

    if (answerKeySelect.value === "custom" && keyFileInput.files.length > 0) {
      formData.append("answer_key_file", keyFileInput.files[0]);
    }

    try {
      const response = await fetch("/api/grade", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errDetail = "Grading request failed.";
        try {
          const errData = await response.json();
          errDetail = errData.detail || errDetail;
        } catch (_) {}
        throw new Error(errDetail);
      }

      const result = await response.json();
      currentResultId = result.result_id;
      renderResults(result);
    } catch (err) {
      showError("Evaluation Error", err.message || "An unexpected error occurred during processing.");
    } finally {
      setLoading(false);
    }
  }

  processBtn.addEventListener("click", triggerGrading);

  // --- Render Results ---
  function renderResults(data) {
    const summary = data.summary || {};
    const questions = data.questions || [];
    const issues = data.flagged_issues || [];

    // 1. Metric Cards
    marksObtainedVal.textContent = (summary.marks_obtained !== undefined) ? summary.marks_obtained.toFixed(2) : "0.00";
    totalMarksVal.textContent = (summary.total_marks !== undefined) ? summary.total_marks.toFixed(2) : "120.00";
    const pct = summary.percentage !== undefined ? summary.percentage : 0;
    percentageVal.textContent = `${pct.toFixed(1)}%`;
    percentageBar.style.width = `${Math.min(100, Math.max(0, pct))}%`;

    correctCountVal.textContent = summary.correct_count !== undefined ? summary.correct_count : 0;
    incorrectCountVal.textContent = summary.incorrect_count !== undefined ? summary.incorrect_count : 0;
    multiMarkCountVal.textContent = summary.multiple_mark_count !== undefined ? summary.multiple_mark_count : 0;
    unansweredCountVal.textContent = summary.unanswered_count !== undefined ? summary.unanswered_count : 0;

    // 2. Student Metadata
    const metadata = data.student_details || {};
    const fields = (metadata.fields && typeof metadata.fields === "object") ? metadata.fields : metadata;

    function renderField(valElem, confElem, fieldData) {
      let text = "";
      let rawConf = 0.0;
      if (typeof fieldData === "string") {
        text = fieldData.trim();
        rawConf = 0.85;
      } else if (fieldData && typeof fieldData === "object") {
        const raw = fieldData.text !== undefined ? fieldData.text : fieldData.extracted_text;
        text = (typeof raw === "string") ? raw.trim() : (raw !== undefined && raw !== null ? String(raw).trim() : "");
        rawConf = (fieldData.confidence !== undefined) ? fieldData.confidence : 0.0;
      }
      if (!text) {
        valElem.textContent = "Not Detected";
        confElem.textContent = "0%";
        confElem.className = "conf-badge conf-low";
        return;
      }
      valElem.textContent = text;
      const conf = Math.round(rawConf * 100);
      confElem.textContent = `${conf}%`;
      if (conf >= 80) {
        confElem.className = "conf-badge conf-high";
      } else if (conf >= 50) {
        confElem.className = "conf-badge conf-mid";
      } else {
        confElem.className = "conf-badge conf-low";
      }
    }

    renderField(studentIdVal, studentIdConf, fields.student_number_written || fields.student_id);
    renderField(studentNameVal, studentNameConf, fields.name);
    renderField(moduleCodeVal, moduleCodeConf, fields.module_code);
    renderField(seatNumberVal, seatNumberConf, fields.seat_number);
    renderField(examDateVal, examDateConf, fields.date || fields.exam_date);

    if (ocrEngineBadge) {
      const enginesUsed = new Set();
      Object.values(fields).forEach((f) => {
        if (f && f.engine && f.engine !== "none" && f.engine !== "failed") {
          enginesUsed.add(f.engine === "trocr" ? "TrOCR" : (f.engine === "easyocr" ? "EasyOCR" : f.engine));
        }
      });
      if (enginesUsed.size > 0) {
        ocrEngineBadge.textContent = Array.from(enginesUsed).join(" / ");
      } else {
        ocrEngineBadge.textContent = "TrOCR / EasyOCR";
      }
    }

    // Alignment Confidence
    const alignConf = data.alignment_confidence !== undefined ? data.alignment_confidence : 0;
    alignmentConfidenceVal.textContent = `${(alignConf * 100).toFixed(1)}%`;
    if (alignConf >= 0.75) {
      alignmentStatusBadge.textContent = "High Alignment";
      alignmentStatusBadge.className = "badge badge-neutral";
    } else if (alignConf >= 0.65) {
      alignmentStatusBadge.textContent = "Fair Alignment";
      alignmentStatusBadge.className = "badge badge-alert";
    } else {
      alignmentStatusBadge.textContent = "LOW_ALIGNMENT_CONFIDENCE";
      alignmentStatusBadge.className = "badge badge-alert";
    }

    // 3. Flagged Issues Panel
    flaggedIssuesList.innerHTML = "";
    if (issues.length === 0) {
      flaggedCountBadge.textContent = "0 Issues";
      flaggedCountBadge.className = "badge badge-neutral";
      flaggedIssuesList.innerHTML = '<div class="empty-issues">No anomalies detected. All markings conform to policy.</div>';
    } else {
      flaggedCountBadge.textContent = `${issues.length} Issues`;
      flaggedCountBadge.className = "badge badge-alert";
      issues.forEach((iss) => {
        const item = document.createElement("div");
        item.className = "issue-item";

        const prefix = iss.question ? `Q${iss.question}` : (iss.target_field || "Sheet");
        const category = iss.category || iss.issue_type || "Issue";
        const message = iss.message || iss.description || "";
        item.innerHTML = `
          <div class="issue-header">
            <span class="badge badge-alert">${escapeHtml(category)}</span>
            <span class="issue-target">${escapeHtml(prefix)}</span>
          </div>
          <p class="issue-desc">${escapeHtml(message)}</p>
        `;
        flaggedIssuesList.appendChild(item);
      });
    }

    // 4. Overlaid Image Preview
    if (data.overlay_preview) {
      overlayPreviewImage.src = data.overlay_preview;
      modalPreviewImage.src = data.overlay_preview;
    }

    // 5. Question Table Breakdown
    questionTableBody.innerHTML = "";
    questions.forEach((q) => {
      const tr = document.createElement("tr");

      const qNum = q.question;
      const selected = q.selected_options && q.selected_options.length > 0
        ? q.selected_options.join(", ")
        : "None";
      const expected = q.expected_answer || "—";
      const statusText = q.status || "—";
      const marks = q.marks_awarded !== undefined ? q.marks_awarded.toFixed(2) : "0.00";

      // Best confidence from options
      let confPercent = "—";
      if (q.confidences && Object.keys(q.confidences).length > 0) {
        const maxConf = Math.max(...Object.values(q.confidences));
        confPercent = `${Math.round(maxConf * 100)}%`;
      }

      // Status badge styling
      let statusClass = "badge-neutral";
      if (statusText === "CORRECT") statusClass = "badge-correct";
      else if (statusText === "INCORRECT") statusClass = "badge-incorrect";
      else if (statusText === "MULTIPLE_MARK") statusClass = "badge-multimark";
      else if (statusText === "REVIEW") statusClass = "badge-review";

      // Flags
      const flagsText = (q.flags && q.flags.length > 0) ? q.flags.join(", ") : "—";

      tr.innerHTML = `
        <td class="font-bold">Q${qNum}</td>
        <td><span class="option-pill ${selected === 'None' ? 'option-none' : ''}">${escapeHtml(selected)}</span></td>
        <td><strong>${escapeHtml(expected)}</strong></td>
        <td><span class="badge ${statusClass}">${escapeHtml(statusText)}</span></td>
        <td class="font-mono ${Number(marks) > 0 ? 'text-success' : (Number(marks) < 0 ? 'text-danger' : '')}">${marks}</td>
        <td>${confPercent}</td>
        <td class="text-xs text-muted">${escapeHtml(flagsText)}</td>
      `;
      questionTableBody.appendChild(tr);
    });

    resultsSection.classList.remove("hidden");
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // --- Modal Zoom Controls ---
  togglePreviewZoomBtn.addEventListener("click", () => {
    previewModal.classList.remove("hidden");
  });

  closeModalBtn.addEventListener("click", () => {
    previewModal.classList.add("hidden");
  });

  modalBackdrop.addEventListener("click", () => {
    previewModal.classList.add("hidden");
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !previewModal.classList.contains("hidden")) {
      previewModal.classList.add("hidden");
    }
  });

  // --- Export Actions ---
  exportCsvBtn.addEventListener("click", () => {
    if (!currentResultId) return;
    window.location.href = `/api/export/${currentResultId}.csv`;
  });

  exportJsonBtn.addEventListener("click", () => {
    if (!currentResultId) return;
    window.location.href = `/api/export/${currentResultId}.json`;
  });

  // --- Utility Helpers ---
  function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
