/**
 * LectureMind AI — Obsidian/ThetaWave Frontend Controller
 */

// ── Application State ─────────────────────────────────────────────────────────
let currentVideoId = null;        // SQLite database row ID
let currentYouTubeId = null;      // 11-char YouTube ID
let currentVideoTitle = null;
let youtubePlayer = null;
let isPlayerReady = false;
let allTranscriptSegments = [];   // Cached transcript segments for live search
let activeTab = "chat";

// Phase 2 state
let currentNotesMode = "summary";
let currentNotesContent = null;
let currentQuizNum = 5;
let currentQuizId = null;
let currentQuizQuestions = [];
let userQuizAnswers = {};         // { "1": 2, "2": 0, ... } questionId → selectedIndex


// ── Auth & User State (ThetaWave Style) ───────────────────────────────────────
// Clean up any legacy localStorage tokens from previous test runs so webpage opens cleanly with NO login
if (typeof localStorage !== "undefined" && localStorage.getItem("lecturemind_auth_token")) {
  localStorage.removeItem("lecturemind_auth_token");
}

let currentUser = null;           // { id, email, name, avatar_color, video_count, chat_count }
let authToken = typeof sessionStorage !== "undefined" ? sessionStorage.getItem("lecturemind_auth_token") : null;
let pendingProcessUrl = null;     // Stored URL to auto-process right after login
let authTabMode = "login";        // "login" | "signup"

function getAuthHeaders() {
  const headers = { "Content-Type": "application/json" };
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }
  return headers;
}


// Suggested prompt starter chips
const DEFAULT_SUGGESTIONS = [
  "Explain the core concept in simple terms",
  "Summarize key formulas and definitions",
  "What examples are given in this video?",
  "Generate 3 practice questions with timestamps",
  "What are the most important exam takeaways?"
];



// ── YouTube Player API ────────────────────────────────────────────────────────
function onYouTubeIframeAPIReady() {
  console.log("[YT API] Ready");
}

function initYouTubePlayer(youtubeVideoId) {
  isPlayerReady = false;

  if (youtubePlayer && typeof youtubePlayer.loadVideoById === "function") {
    try {
      youtubePlayer.loadVideoById(youtubeVideoId);
      isPlayerReady = true;
      return;
    } catch (e) {
      console.warn("Re-creating player:", e);
    }
  }

  try {
    youtubePlayer = new YT.Player("youtube-player", {
      videoId: youtubeVideoId,
      playerVars: {
        autoplay: 0,
        modestbranding: 1,
        rel: 0,
        origin: window.location.origin,
      },
      events: {
        onReady: () => {
          isPlayerReady = true;
          console.log("[YT Player] Ready for seeking");
        },
        onError: (e) => console.warn("[YT Player] Error:", e.data)
      }
    });
  } catch (err) {
    console.error("Failed to initialize YT Player:", err);
  }
}

function seekTo(seconds) {
  const sec = parseFloat(seconds);
  if (youtubePlayer && isPlayerReady && typeof youtubePlayer.seekTo === "function") {
    youtubePlayer.seekTo(sec, true);
    youtubePlayer.playVideo();
    showToast(`Jumped to ${formatTime(sec)}`, "info");
  } else {
    window.open(`https://www.youtube.com/watch?v=${currentYouTubeId}&t=${Math.floor(sec)}s`, "_blank");
  }
}

function seekRelative(offset) {
  if (youtubePlayer && isPlayerReady && typeof youtubePlayer.getCurrentTime === "function") {
    const current = youtubePlayer.getCurrentTime();
    seekTo(Math.max(0, current + offset));
  }
}


// ── Auth Manager (ThetaWave Style) ────────────────────────────────────────────

async function initAuthSession() {
  if (authToken) {
    try {
      const res = await fetch("/api/auth/me", { headers: getAuthHeaders() });
      if (res.ok) {
        currentUser = await res.json();
        renderUserWidget(currentUser);
      } else {
        // Token invalid / expired
        authToken = null;
        currentUser = null;
        if (typeof sessionStorage !== "undefined") sessionStorage.removeItem("lecturemind_auth_token");
        if (typeof localStorage !== "undefined") localStorage.removeItem("lecturemind_auth_token");
        renderUserWidget(null);

      }
    } catch (e) {
      console.warn("[Auth] Failed to verify token:", e);
      renderUserWidget(null);
    }
  } else {
    renderUserWidget(null);
  }
  loadLibrary();
}

function renderUserWidget(user) {
  const loggedOutBox = document.getElementById("user-widget-logged-out");
  const loggedInBox = document.getElementById("user-widget-logged-in");

  if (!loggedOutBox || !loggedInBox) return;

  if (user) {
    hideElement(loggedOutBox);
    showElement(loggedInBox);

    // Compute initials from name (e.g. "Aditya Mali" -> "AM")
    const names = (user.name || "User").trim().split(" ");
    const initials = names.length > 1
      ? (names[0][0] + names[names.length - 1][0]).toUpperCase()
      : names[0].slice(0, 2).toUpperCase();

    const avatarEl = document.getElementById("sidebar-user-avatar");
    if (avatarEl) {
      avatarEl.textContent = initials;
      avatarEl.style.background = user.avatar_color || "#6366F1";
    }

    const nameEl = document.getElementById("sidebar-user-name");
    if (nameEl) nameEl.textContent = user.name;

    const emailEl = document.getElementById("sidebar-user-email");
    if (emailEl) emailEl.textContent = user.email;

    const dropNameEl = document.getElementById("dropdown-user-name");
    if (dropNameEl) dropNameEl.textContent = user.name;

    const dropEmailEl = document.getElementById("dropdown-user-email");
    if (dropEmailEl) dropEmailEl.textContent = user.email;

    const dropStatsEl = document.getElementById("dropdown-stats");
    if (dropStatsEl) {
      dropStatsEl.textContent = `⚡ ${user.video_count || 0} lectures in library`;
    }
  } else {
    showElement(loggedOutBox);
    hideElement(loggedInBox);
    closeUserDropdown();
  }
}

function openAuthModal(mode = "login", customSubtitle = null) {
  authTabMode = mode;
  switchAuthTab(mode);

  const subEl = document.getElementById("auth-modal-subtitle");
  if (subEl) {
    subEl.textContent = customSubtitle || "Save your study history, personalized notes, and interactive timestamped Q&A.";
  }

  const errBanner = document.getElementById("auth-error-banner");
  if (errBanner) hideElement(errBanner);

  // Clear password
  const passInput = document.getElementById("auth-password-input");
  if (passInput) passInput.value = "";

  showElement(document.getElementById("auth-modal"));
}

function closeAuthModal() {
  hideElement(document.getElementById("auth-modal"));
}

function switchAuthTab(mode) {
  authTabMode = mode;
  const loginTab = document.getElementById("auth-tab-login");
  const signupTab = document.getElementById("auth-tab-signup");
  const nameField = document.getElementById("auth-name-field");
  const submitText = document.getElementById("auth-submit-text");
  const modalTitle = document.getElementById("auth-modal-title");

  if (mode === "signup") {
    if (loginTab) loginTab.classList.remove("active");
    if (signupTab) signupTab.classList.add("active");
    if (nameField) nameField.style.display = "block";
    if (submitText) submitText.textContent = "Create Account";
    if (modalTitle) modalTitle.textContent = "Create Your Account";
  } else {
    if (loginTab) loginTab.classList.add("active");
    if (signupTab) signupTab.classList.remove("active");
    if (nameField) nameField.style.display = "none";
    if (submitText) submitText.textContent = "Sign In";
    if (modalTitle) modalTitle.textContent = "Sign In to LectureMind";
  }
}

async function handleAuthSubmit(event) {
  event.preventDefault();
  const errBanner = document.getElementById("auth-error-banner");
  const submitBtn = document.getElementById("auth-submit-btn");

  const email = document.getElementById("auth-email-input").value.trim();
  const password = document.getElementById("auth-password-input").value;
  const name = document.getElementById("auth-name-input") ? document.getElementById("auth-name-input").value.trim() : "";

  if (authTabMode === "signup" && !name) {
    showBannerError("auth-error-banner", "Please enter your full name.");
    return;
  }

  if (submitBtn) submitBtn.disabled = true;
  hideElement(errBanner);

  try {
    const endpoint = authTabMode === "signup" ? "/api/auth/signup" : "/api/auth/login";
    const body = authTabMode === "signup"
      ? { email, password, name }
      : { email, password };

    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.detail || "Authentication failed.");
    }

    // Success
    authToken = data.token;
    currentUser = data.user;
    if (typeof sessionStorage !== "undefined") {
      sessionStorage.setItem("lecturemind_auth_token", authToken);
    }

    renderUserWidget(currentUser);
    closeAuthModal();
    loadLibrary();
    showToast(data.message || "Signed in successfully!", "info");

    // If user clicked "Analyze Lecture" before logging in, proceed automatically now!
    if (pendingProcessUrl) {
      const urlToRun = pendingProcessUrl;
      pendingProcessUrl = null;
      document.getElementById("youtube-url-input").value = urlToRun;
      executeVideoProcessing(urlToRun, "hero-process-btn", "onboarding-loader", "onboarding-error");
    }

  } catch (err) {
    showBannerError("auth-error-banner", err.message);
  } finally {
    if (submitBtn) submitBtn.disabled = false;
  }
}

async function handleGuestLogin() {
  const errBanner = document.getElementById("auth-error-banner");
  hideElement(errBanner);

  try {
    const res = await fetch("/api/auth/guest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Guest sign in failed.");

    authToken = data.token;
    currentUser = data.user;
    if (typeof sessionStorage !== "undefined") {
      sessionStorage.setItem("lecturemind_auth_token", authToken);
    }

    renderUserWidget(currentUser);
    closeAuthModal();
    loadLibrary();
    showToast("Signed in as Demo Student!", "info");

    if (pendingProcessUrl) {
      const urlToRun = pendingProcessUrl;
      pendingProcessUrl = null;
      document.getElementById("youtube-url-input").value = urlToRun;
      executeVideoProcessing(urlToRun, "hero-process-btn", "onboarding-loader", "onboarding-error");
    }
  } catch (err) {
    showBannerError("auth-error-banner", err.message);
  }
}

function handleLogout() {
  authToken = null;
  currentUser = null;
  if (typeof sessionStorage !== "undefined") {
    sessionStorage.removeItem("lecturemind_auth_token");
  }
  if (typeof localStorage !== "undefined") {
    localStorage.removeItem("lecturemind_auth_token");
    localStorage.removeItem("lecturemind_active_video_id");
  }

  closeUserDropdown();
  renderUserWidget(null);
  loadLibrary();
  goToMainPage();
  showToast("Logged out successfully.", "info");
}


function toggleUserDropdown(event) {
  if (event) event.stopPropagation();
  const dropdown = document.getElementById("user-dropdown-menu");
  if (dropdown) dropdown.classList.toggle("hidden");
}

function closeUserDropdown() {
  const dropdown = document.getElementById("user-dropdown-menu");
  if (dropdown) dropdown.classList.add("hidden");
}

// Click outside to close user dropdown
document.addEventListener("click", (e) => {
  const section = document.getElementById("sidebar-user-section");
  if (section && !section.contains(e.target)) {
    closeUserDropdown();
  }
});


// ── Video Ingestion & Processing ──────────────────────────────────────────────

async function handleProcessVideo() {
  const input = document.getElementById("youtube-url-input");
  const url = input.value.trim();
  if (!url) {
    showBannerError("onboarding-error", "Please paste a valid YouTube lecture URL.");
    return;
  }

  // Gatekeeper: Prompt login if not authenticated
  if (!currentUser) {
    pendingProcessUrl = url;
    openAuthModal("signup", "Please sign in or create an account to process this lecture and save your study notes.");
    return;
  }

  await executeVideoProcessing(url, "hero-process-btn", "onboarding-loader", "onboarding-error");
}

async function submitModalVideo() {
  const input = document.getElementById("modal-url-input");
  const url = input.value.trim();
  if (!url) {
    showBannerError("modal-error", "Please paste a YouTube URL.");
    return;
  }
  closeNewVideoModal();

  // Gatekeeper: Prompt login if not authenticated
  if (!currentUser) {
    pendingProcessUrl = url;
    openAuthModal("signup", "Please sign in or create an account to process this lecture.");
    return;
  }

  switchMainView("onboarding");
  document.getElementById("youtube-url-input").value = url;
  await handleProcessVideo();
}

function loadPresetUrl(url) {
  document.getElementById("youtube-url-input").value = url;
  if (!currentUser) {
    pendingProcessUrl = url;
    openAuthModal("login", "Sign in to study this preset lecture and save your personal notes.");
    return;
  }
  handleProcessVideo();
}

async function executeVideoProcessing(url, btnId, loaderId, errorId) {
  const btn = document.getElementById(btnId);
  const loader = document.getElementById(loaderId);
  const errorEl = document.getElementById(errorId);

  hideElement(errorEl);
  showElement(loader);
  if (btn) btn.disabled = true;

  try {
    const response = await fetch("/api/videos/process", {
      method: "POST",
      headers: getAuthHeaders(),
      body: JSON.stringify({ youtube_url: url }),
    });

    const data = await response.json();

    if (!response.ok) {
      if (response.status === 401) {
        pendingProcessUrl = url;
        openAuthModal("login", "Your session expired. Please sign in again to process lectures.");
        return;
      }
      throw new Error(data.detail || "Video processing failed.");
    }

    // Processing Success!
    currentVideoId = String(data.video_id);
    currentYouTubeId = data.youtube_video_id;
    currentVideoTitle = data.title || `Lecture (${currentYouTubeId})`;

    localStorage.setItem("lecturemind_active_video_id", String(data.video_id));

    updateUIForLoadedVideo(data);
    switchMainView("workspace");
    initYouTubePlayer(currentYouTubeId);
    renderSuggestions(DEFAULT_SUGGESTIONS);
    fetchAndDisplayTranscript(currentVideoId);
    fetchAndDisplayChatHistory(currentVideoId);
    fetchAndDisplaySavedNotes(currentVideoId);

    // Refresh the library and user stats
    loadLibrary();
    if (currentUser) {
      currentUser.video_count = (currentUser.video_count || 0) + 1;
      renderUserWidget(currentUser);
    }

    if (data.from_cache) {
      showToast(`⚡ Loaded from cache — 0 embedding API calls used!`, "info");
    } else {
      showToast(`✅ Successfully indexed ${data.chunk_count} lecture chunks!`);
    }

  } catch (err) {
    showBannerError(errorId, err.message);
  } finally {
    hideElement(loader);
    if (btn) btn.disabled = false;
  }
}

// ── Cached Library State ──────────────────────────────────────────────────────
let cachedLibraryVideos = [];

function updateUIForLoadedVideo(data) {
  const title = data.title || "YouTube Lecture";
  const topbarTitle = document.getElementById("topbar-title");
  if (topbarTitle) topbarTitle.textContent = title;
  const lectureTitle = document.getElementById("lecture-title-display");
  if (lectureTitle) lectureTitle.textContent = title;
  const sidebarTitle = document.getElementById("sidebar-video-title");
  if (sidebarTitle) sidebarTitle.textContent = title;
  const sidebarChunks = document.getElementById("sidebar-video-chunks");
  if (sidebarChunks) sidebarChunks.textContent = `${data.chunk_count} indexed chunks`;
  const chunkBadge = document.getElementById("chunk-count-badge");
  if (chunkBadge) chunkBadge.textContent = `${data.chunk_count} Chunks`;

  const notesTitle = document.getElementById("notes-lecture-title");
  if (notesTitle) notesTitle.textContent = title;
  const quizTitle = document.getElementById("quiz-lecture-title");
  if (quizTitle) quizTitle.textContent = title;
  const durationPill = document.getElementById("notes-duration-pill");
  if (durationPill && data.duration) durationPill.textContent = `⏱️ ${secToTs(data.duration)}`;

  // Highlight the active item in the sidebar library
  document.querySelectorAll(".library-item").forEach(item => {
    item.classList.toggle("active", item.dataset.videoId === String(data.video_id));
  });
}



// ── Library & Navigation ──────────────────────────────────────────────────────

function goToMainPage() {
  switchMainView("onboarding");
  const input = document.getElementById("youtube-url-input");
  if (input) input.focus();
}

async function loadLibrary() {
  try {
    const res = await fetch("/api/videos", { headers: getAuthHeaders() });
    if (!res.ok) {
      if (res.status === 401) {
        cachedLibraryVideos = [];
        renderLibraryUI([]);
      }
      return;
    }
    const videos = await res.json();
    cachedLibraryVideos = Array.isArray(videos) ? videos : [];

    renderLibraryUI(cachedLibraryVideos);
  } catch (e) {
    console.warn("[Library] Failed to load:", e);
  }
}



function renderLibraryUI(videos) {
  const list = document.getElementById("library-list");
  const emptyEl = document.getElementById("library-empty");
  const countBadge = document.getElementById("library-count-badge");

  if (!list) return;

  if (!videos || videos.length === 0) {
    list.innerHTML = "";
    if (emptyEl) {
      list.appendChild(emptyEl);
      emptyEl.style.display = "flex";
    }
    if (countBadge) countBadge.textContent = "";
    return;
  }

  if (countBadge) countBadge.textContent = videos.length;

  list.innerHTML = videos.map(v => {
    const isActive = String(v.video_id) === String(currentVideoId);
    const date = v.processed_at
      ? new Date(v.processed_at).toLocaleDateString("en-IN", { month: "short", day: "numeric" })
      : "";
    return `
      <div class="library-item ${isActive ? "active" : ""}"
           data-video-id="${v.video_id}"
           onclick="handleLibraryCardClick('${v.video_id}')"
           title="${escapeHtml(v.title || "Lecture")}">
        <div class="library-item-thumb">
          <img src="https://img.youtube.com/vi/${v.youtube_video_id}/default.jpg"
               onerror="this.style.display='none'"
               alt="" />
        </div>
        <div class="library-item-meta">
          <div class="library-item-title">${escapeHtml(v.title || "Untitled")}</div>
          <div class="library-item-sub">
            <span>${v.chunk_count} chunks</span>
            ${date ? `<span>·</span><span>${date}</span>` : ""}
          </div>
        </div>
      </div>
    `;
  }).join("");
}

function handleLibraryCardClick(videoId) {

  const video = cachedLibraryVideos.find(v => String(v.video_id) === String(videoId));
  if (video) {
    loadVideoFromLibrary(video.video_id, video.youtube_video_id, video.title, video.chunk_count);
  }
}

async function loadVideoFromLibrary(videoId, youtubeId, title, chunkCount, isAutoRestore = false) {
  // If already active in workspace, just switch view
  if (String(videoId) === String(currentVideoId) && activeTab) {
    switchMainView("workspace");
    return;
  }

  currentVideoId = String(videoId);
  currentYouTubeId = youtubeId;
  currentVideoTitle = title || `Lecture (${youtubeId})`;

  // Save to localStorage for automatic reload persistence
  localStorage.setItem("lecturemind_active_video_id", String(videoId));

  updateUIForLoadedVideo({ video_id: videoId, title: currentVideoTitle, chunk_count: chunkCount });
  switchMainView("workspace");
  initYouTubePlayer(youtubeId);
  renderSuggestions(DEFAULT_SUGGESTIONS);

  // Parallel fetch: Transcript + Chat History + Saved Notes
  fetchAndDisplayTranscript(videoId);
  fetchAndDisplayChatHistory(videoId);
  fetchAndDisplaySavedNotes(videoId);

  if (!isAutoRestore) {
    showToast(`⚡ Loaded "${currentVideoTitle}" (0 API calls)`, "info");
  }
}

function renderChatWelcomeState(title, chunkCount) {
  const container = document.getElementById("chat-messages-container");
  if (!container) return;

  const lectureTitle = title || currentVideoTitle || "Current Lecture";
  const safeTitle = escapeHtml(lectureTitle);
  const chunks = chunkCount || (allTranscriptSegments ? allTranscriptSegments.length : 0);
  const chunkText = chunks > 0 ? `<strong>${chunks} indexed chunks</strong>` : "complete lecture transcript";

  container.innerHTML = `
    <div class="chat-welcome-card">
      <div class="welcome-header">
        <div class="welcome-badge">
          <span class="welcome-sparkle">✨</span>
          <span>AI Lecture Tutor Ready</span>
        </div>
        <h3 class="welcome-title">Ask anything about this lecture</h3>
        <p class="welcome-subtitle">
          I've indexed the ${chunkText} for <em>"${safeTitle}"</em> with exact timestamp citations. Ask a question, or pick a study action below:
        </p>
      </div>

      <div class="welcome-cards-grid">
        <div class="welcome-action-card" onclick="askQuickPrompt('Explain the core concept and main idea of this lecture in simple terms')">
          <div class="action-card-header">
            <span class="action-card-icon">💡</span>
            <span class="action-card-title">Core Concept Breakdown</span>
          </div>
          <p class="action-card-desc">Get a clear, beginner-friendly explanation of what this lecture teaches.</p>
        </div>

        <div class="welcome-action-card" onclick="askQuickPrompt('Summarize all key formulas, terms, and definitions with timestamps')">
          <div class="action-card-header">
            <span class="action-card-icon">📝</span>
            <span class="action-card-title">Key Definitions & Formulas</span>
          </div>
          <p class="action-card-desc">Quick reference list of critical terminology and formulas.</p>
        </div>

        <div class="welcome-action-card" onclick="askQuickPrompt('What are the code examples or practical steps demonstrated in this video?')">
          <div class="action-card-header">
            <span class="action-card-icon">💻</span>
            <span class="action-card-title">Code & Implementations</span>
          </div>
          <p class="action-card-desc">Review step-by-step code snippets and practical demos.</p>
        </div>

        <div class="welcome-action-card" onclick="askQuickPrompt('Give me 3 important interview and exam questions based on this video')">
          <div class="action-card-header">
            <span class="action-card-icon">🎯</span>
            <span class="action-card-title">Exam & Interview Prep</span>
          </div>
          <p class="action-card-desc">Test your understanding with targeted questions from the lecture.</p>
        </div>
      </div>
    </div>
  `;
}

async function fetchAndDisplayChatHistory(videoId) {
  const container = document.getElementById("chat-messages-container");
  if (!container) return;

  try {
    const res = await fetch(`/api/chat/history/${videoId}`);
    if (!res.ok) {
      renderChatWelcomeState();
      return;
    }
    const data = await res.json();

    container.innerHTML = ""; // Clear existing

    if (data.history && data.history.length > 0) {
      data.history.forEach(item => {
        appendChatBubble(item.question, "user");
        appendChatBubble(item.answer, "bot", item.cited_timestamps || []);
      });
    } else {
      renderChatWelcomeState();
    }
  } catch (e) {
    console.warn("[Chat History] Could not load past history:", e);
    renderChatWelcomeState();
  }
}


async function fetchAndDisplaySavedNotes(videoId) {
  try {
    const res = await fetch(`/api/notes/${videoId}?mode=${currentNotesMode}`);
    if (!res.ok) return;
    const data = await res.json();

    if (data && data.content) {
      currentNotesContent = data.content;
      renderNotesOutput(data.content, data.mode);
    } else {
      // No notes yet — reset to empty state
      hideElement(document.getElementById("notes-output"));
      showElement(document.getElementById("notes-empty"));
      const wrap = document.getElementById("download-notes-wrap");
      if (wrap) wrap.style.display = "none";
      currentNotesContent = null;
    }
  } catch (e) {
    console.warn("[Notes] Could not load saved notes:", e);
  }
}





async function fetchAndDisplayTranscript(videoId) {
  const container = document.getElementById("transcript-segments-container");
  const rawContainer = document.getElementById("raw-transcript-text-container");

  try {
    const res = await fetch(`/api/videos/${videoId}/chunks`);
    if (!res.ok) return;
    const data = await res.json();

    allTranscriptSegments = data.chunks || [];

    if (allTranscriptSegments.length === 0) {
      container.innerHTML = `<div class="transcript-loading-state">No transcript chunks found.</div>`;
      return;
    }

    container.innerHTML = allTranscriptSegments.map(chunk => `
      <div class="transcript-segment-row" onclick="seekTo(${chunk.start_time})">
        <button class="time-pill-btn" title="Seek to ${formatTime(chunk.start_time)}">
          ▶ ${formatTime(chunk.start_time)}
        </button>
        <span class="segment-text">${escapeHtml(chunk.text)}</span>
      </div>
    `).join("");

    if (rawContainer && data.raw_transcript) {
      rawContainer.innerHTML = `<p>${escapeHtml(data.raw_transcript)}</p>`;
    }
  } catch (e) {
    console.error("Could not load transcript segments", e);
  }
}

function filterTranscript(keyword) {
  const term = keyword.toLowerCase().trim();
  const rows = document.querySelectorAll(".transcript-segment-row");
  rows.forEach(row => {
    const text = row.innerText.toLowerCase();
    row.style.display = text.includes(term) ? "flex" : "none";
  });
}


// ── AI Chat ────────────────────────────────────────────────────────────────────
async function sendChatMessage() {
  const textarea = document.getElementById("chat-textarea");
  const sendBtn = document.getElementById("send-message-btn");
  const question = textarea.value.trim();

  if (!question || !currentVideoId) return;

  textarea.value = "";
  sendBtn.disabled = true;

  // Append user bubble
  appendChatBubble(question, "user");

  // Show animated bot typing indicator
  const typingId = showTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: currentVideoId,
        question: question,
      }),
    });

    const data = await response.json();
    removeTypingIndicator(typingId);

    if (!response.ok) {
      appendChatBubble(`⚠️ ${data.detail || "Error generating response."}`, "bot", []);
      return;
    }

    appendChatBubble(data.answer, "bot", data.cited_timestamps || []);

  } catch (err) {
    removeTypingIndicator(typingId);
    appendChatBubble("⚠️ Network error. Please check your connection and try again.", "bot", []);
  } finally {
    sendBtn.disabled = false;
    textarea.focus();
  }
}

function askQuickPrompt(promptText) {
  switchMainTab("chat");
  document.getElementById("chat-textarea").value = promptText;
  sendChatMessage();
}

function handleChatKeyDown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendChatMessage();
  }
}

function appendChatBubble(text, role, citations = []) {
  const container = document.getElementById("chat-messages-container");
  if (!container) return;

  const welcomeCard = container.querySelector(".chat-welcome-card");
  if (welcomeCard) {
    welcomeCard.remove();
  }

  const row = document.createElement("div");
  row.className = `chat-bubble-row ${role === "user" ? "user-row" : "bot-row"}`;


  if (role === "bot") {
    row.innerHTML = `
      <div class="bot-avatar"><span>✨</span></div>
      <div class="bubble-content bot-bubble">
        <div class="bot-author">LectureMind AI</div>
        <div class="bubble-markdown">${formatMarkdown(text)}</div>
        ${renderCitationsHtml(citations)}
      </div>
    `;
  } else {
    row.innerHTML = `
      <div class="bubble-content user-bubble">
        ${escapeHtml(text)}
      </div>
    `;
  }

  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function renderCitationsHtml(citations) {
  if (!citations || citations.length === 0) return "";

  const pills = citations.map(c => `
    <button class="citation-pill" onclick="seekTo(${c.start})" title="Jump to ${c.label} in video">
      <span class="citation-play-icon">▶</span>
      <span>${escapeHtml(c.label)}</span>
    </button>
  `).join("");

  return `<div class="citations-box">${pills}</div>`;
}

function showTypingIndicator() {
  const container = document.getElementById("chat-messages-container");
  const id = `typing-${Date.now()}`;
  const row = document.createElement("div");
  row.className = "chat-bubble-row bot-row";
  row.id = id;
  row.innerHTML = `
    <div class="bot-avatar"><span>✨</span></div>
    <div class="bubble-content bot-bubble">
      <div class="typing-dots">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>
  `;
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}


// ── Tab & View Navigation ──────────────────────────────────────────────────────
function switchMainView(view) {
  const onboarding = document.getElementById("onboarding-view");
  const workspace = document.getElementById("study-workspace");

  if (view === "workspace") {
    hideElement(onboarding);
    showElement(workspace);
  } else {
    showElement(onboarding);
    hideElement(workspace);
  }
}

function switchMainTab(tabKey) {
  activeTab = tabKey;

  const workspace = document.getElementById("study-workspace");
  if (workspace) {
    workspace.classList.remove("mode-chat", "mode-notes", "mode-quiz", "mode-transcript");
    workspace.classList.add(`mode-${tabKey}`);
  }

  // Update sidebar nav items
  document.querySelectorAll(".sidebar-nav .nav-item").forEach(item => item.classList.remove("active"));
  const activeNav = document.getElementById(`nav-${tabKey}`);
  if (activeNav) activeNav.classList.add("active");

  // Update panels
  document.querySelectorAll(".tab-content-panel").forEach(panel => panel.classList.remove("active"));
  const targetPanel = document.getElementById(`panel-${tabKey}`);
  if (targetPanel) targetPanel.classList.add("active");

  // If notes opened, ensure saved notes are loaded if not yet present
  if (tabKey === "notes") {
    const notesCanvas = document.querySelector(".dedicated-notes-canvas");
    if (notesCanvas) notesCanvas.scrollTop = 0;
    if (currentVideoId && !currentNotesContent) {
      fetchAndDisplaySavedNotes(currentVideoId);
    }
  } else if (tabKey === "quiz") {
    const quizCanvas = document.querySelector(".dedicated-quiz-canvas");
    if (quizCanvas) quizCanvas.scrollTop = 0;
  }
}


function renderSuggestions(suggestions) {
  const container = document.getElementById("suggestion-chips-container");
  if (!container) return;
  container.innerHTML = "";

  suggestions.forEach(text => {
    const pill = document.createElement("button");
    pill.className = "suggestion-pill";
    pill.textContent = text;
    pill.onclick = () => {
      document.getElementById("chat-textarea").value = text;
      sendChatMessage();
    };
    container.appendChild(pill);
  });
}


// ── Modal & UI Helpers ─────────────────────────────────────────────────────────
function openNewVideoModal() {
  showElement(document.getElementById("new-video-modal"));
  const input = document.getElementById("modal-url-input");
  input.value = "";
  input.focus();
}

function closeNewVideoModal() {
  hideElement(document.getElementById("new-video-modal"));
}

function toggleSidebar() {
  const sidebar = document.getElementById("sidebar");
  sidebar.classList.toggle("collapsed");
}

function showBannerError(elId, msg) {
  const el = document.getElementById(elId);
  if (el) {
    el.textContent = msg;
    showElement(el);
  }
}

function showToast(message, type = "info") {
  const toast = document.getElementById("toast-notification");
  toast.textContent = message;
  showElement(toast);
  setTimeout(() => hideElement(toast), 3500);
}

function showElement(el) {
  if (el) el.classList.remove("hidden");
}

function hideElement(el) {
  if (el) el.classList.add("hidden");
}

function formatTime(seconds) {
  const s = Math.floor(seconds);
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}:${rem < 10 ? "0" : ""}${rem}`;
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function formatMarkdown(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/^[\*\-] (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`)
    .replace(/\n/g, "<br>");
}

// ── Phase 2: Smart Notes ───────────────────────────────────────────────────────

function setNotesMode(mode) {
  currentNotesMode = mode;
  document.querySelectorAll("#notes-mode-toggle .mode-pill").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  if (currentVideoId) {
    fetchAndDisplaySavedNotes(currentVideoId);
  }
}


async function generateNotes() {
  if (!currentVideoId) {
    showToast("Please load a video first.", "error");
    return;
  }

  // Show loading, hide others
  showElement(document.getElementById("notes-loading"));
  hideElement(document.getElementById("notes-empty"));
  hideElement(document.getElementById("notes-output"));
  document.getElementById("generate-notes-btn").disabled = true;
  document.getElementById("download-notes-wrap").style.display = "none";

  try {
    const resp = await fetch("/api/notes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: String(currentVideoId), mode: currentNotesMode }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Notes generation failed.");
    }

    const data = await resp.json();
    currentNotesContent = data.content;

    // Render Markdown
    renderNotesOutput(data.content, data.mode);
    showToast(`${data.mode === "summary" ? "Summary" : "Detailed"} notes generated!`, "info");

  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    showElement(document.getElementById("notes-empty"));
  } finally {
    hideElement(document.getElementById("notes-loading"));
    document.getElementById("generate-notes-btn").disabled = false;
  }
}

function renderNotesOutput(markdownContent, mode) {
  const body = document.getElementById("notes-content-body");
  const badge = document.getElementById("notes-mode-badge");
  const tsLabel = document.getElementById("notes-generated-at");
  const outputCard = document.getElementById("notes-output");
  const downloadWrap = document.getElementById("download-notes-wrap");

  badge.textContent = mode === "summary" ? "Quick Summary" : "Detailed Notes";
  badge.className = `notes-mode-badge ${mode === "detailed" ? "badge-detailed" : ""}`;
  tsLabel.textContent = `Generated ${new Date().toLocaleTimeString()}`;

  body.innerHTML = renderMarkdownFull(markdownContent);

  hideElement(document.getElementById("notes-empty"));
  showElement(outputCard);
  downloadWrap.style.display = "flex";
}

function toggleDownloadMenu() {
  const menu = document.getElementById("download-menu");
  menu.style.display = menu.style.display === "none" ? "block" : "none";
}

// Close the dropdown when clicking outside
document.addEventListener("click", function(e) {
  const wrap = document.getElementById("download-notes-wrap");
  if (wrap && !wrap.contains(e.target)) {
    const menu = document.getElementById("download-menu");
    if (menu) menu.style.display = "none";
  }
});

async function downloadNotes(format = "md") {
  if (!currentNotesContent) return;

  const safeName = (currentVideoTitle || "lecture").replace(/[^a-z0-9]/gi, "_").toLowerCase();

  // Close dropdown
  const menu = document.getElementById("download-menu");
  if (menu) menu.style.display = "none";

  // ── Markdown: client-side blob download ────────────────────────────────────
  if (format === "md") {
    const filename = `${safeName}_${currentNotesMode}_notes.md`;
    const blob = new Blob([currentNotesContent], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Downloaded ${filename}`, "info");
    return;
  }

  // ── PDF / DOCX: server-side generation ────────────────────────────────────
  showToast(`Generating ${format.toUpperCase()}…`, "info");
  try {
    const resp = await fetch("/api/notes/download", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        content: currentNotesContent,
        title: currentVideoTitle || "Lecture Notes",
        format: format,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Download failed.");
    }

    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${safeName}_${currentNotesMode}_notes.${format}`;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Downloaded as ${format.toUpperCase()}!`, "info");
  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
  }
}

// Convert "M:SS" timestamp string to seconds
function tsToSec(tsStr) {
  const parts = tsStr.split(":").map(Number);
  return parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0] * 3600 + parts[1] * 60 + parts[2];
}

// Simple, robust syntax highlighting for lecture code blocks
function highlightPythonSyntax(rawCode) {
  let escaped = escapeHtml(rawCode);

  // Strings (quoted text)
  escaped = escaped.replace(/(["'])(?:(?=(\\?))\2.)*?\1/g, '<span class="tok-str">$&</span>');

  // Keywords
  const kwRegex = /\b(from|import|def|class|return|if|elif|else|for|while|in|as|with|try|except|finally|raise|yield|async|await|lambda|pass|break|continue|True|False|None|self|print)\b/g;
  escaped = escaped.replace(kwRegex, '<span class="tok-kw">$&</span>');

  // Comments (# ...) per line
  const lines = escaped.split("\n");
  const processed = lines.map(line => {
    const hashIdx = line.indexOf("#");
    if (hashIdx >= 0 && !line.slice(0, hashIdx).includes('class="tok-')) {
      const codePart = line.slice(0, hashIdx);
      const commentPart = line.slice(hashIdx).replace(/<[^>]+>/g, "");
      return codePart + `<span class="tok-comment">${commentPart}</span>`;
    }
    return line;
  });

  return processed.join("\n");
}

// Full Markdown renderer (supports headings, bold, italic, lists, tables, fenced code, hr)
function renderMarkdownFull(text) {
  // ── 0. Extract and protect ALL LaTeX math BEFORE any HTML escaping ─────────
  // We stash math regions into placeholder tokens so they survive escaping.
  const mathStash = [];

  function stashMath(raw, displayMode) {
    const idx = mathStash.length;
    mathStash.push({ raw, displayMode });
    return `%%MATH_${idx}%%`;
  }

  // Protect display math first ($$...$$) — greedy multi-line, no HTML escape
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_, inner) => stashMath(inner.trim(), true));

  // Protect inline math ($...$) — single-line only, not empty
  text = text.replace(/\$([^\n$]+?)\$/g, (_, inner) => stashMath(inner.trim(), false));

  // ── Now safe to escape remaining HTML ──────────────────────────────────────
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // ── 1. Fenced code blocks (```lang ... ```) — must run BEFORE inline rules ──
  let codeBlockCounter = 0;
  html = html.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    codeBlockCounter++;
    const blockId = `notes-code-${Date.now()}-${codeBlockCounter}`;
    const displayLang = (lang || "python").toUpperCase();
    const cleanCode = code.trim();
    const highlightedCode = (lang.toLowerCase() === "python" || !lang)
      ? highlightPythonSyntax(cleanCode)
      : escapeHtml(cleanCode);

    return `<div class="notes-code-block" id="${blockId}">` +
      `<div class="code-block-header">` +
        `<span class="code-lang-label">💻 ${displayLang}</span>` +
        `<button class="code-copy-btn" onclick="copyCodeBlock('${blockId}')" title="Copy code">` +
          `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">` +
            `<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>` +
            `<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>` +
          `</svg>` +
          `<span>Copy Code</span>` +
        `</button>` +
      `</div>` +
      `<pre class="code-pre"><code class="code-content">${highlightedCode}</code></pre>` +
    `</div>`;
  });


  // ── 2. Headings ──
  html = html
    .replace(/^#### (.+)$/gm, "<h4>$1</h4>")
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm,  "<h2>$1</h2>")
    .replace(/^# (.+)$/gm,   "<h1>$1</h1>");

  // ── 3. Inline formatting ──
  html = html
    .replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>")
    .replace(/\*\*(.+?)\*\*/g,     "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g,         "<em>$1</em>")
    .replace(/`([^`]+)`/g,         "<code>$1</code>");

  // ── 4. Horizontal rules ──
  html = html.replace(/^---$/gm, "<hr>");

  // ── 5. Tables ──
  html = html.replace(/((?:^\|.+\|\n?)+)/gm, (block) => {
    const rows = block.trim().split("\n");
    if (rows.length < 2) return block;

    let tableHtml = '<table class="notes-table">';
    let inBody = false;

    rows.forEach((row, i) => {
      if (/^\|[-:\s|]+\|$/.test(row.trim())) {
        if (!inBody) { tableHtml += "<tbody>"; inBody = true; }
        return;
      }
      const cells = row.trim().replace(/^\||\|$/g, "").split("|").map(c => c.trim());
      if (i === 0) {
        tableHtml += `<thead><tr>${cells.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;
      } else {
        if (!inBody) { tableHtml += "<tbody>"; inBody = true; }
        tableHtml += `<tr>${cells.map(c => `<td>${c}</td>`).join("")}</tr>`;
      }
    });

    if (inBody) tableHtml += "</tbody>";
    tableHtml += "</table>";
    return tableHtml;
  });

  // ── 6. Blockquotes (> text) ──
  html = html.replace(/((?:^&gt;.+\n?)+)/gm, (block) => {
    const inner = block
      .split("\n")
      .filter(l => l.trim())
      .map(l => l.replace(/^&gt;\s?/, ""))
      .join(" ");
    return `<blockquote>${inner}</blockquote>`;
  });

  // ── 7. Lists ──
  html = html
    .replace(/^\d+\. (.+)$/gm, "<li class='ordered'>$1</li>")
    .replace(/^[*\-] (.+)$/gm,  "<li>$1</li>")
    .replace(/((?:<li[^>]*>.*<\/li>\n?)+)/g, m => {
      const isOrdered = m.includes("class='ordered'");
      const tag = isOrdered ? "ol" : "ul";
      return `<${tag}>${m}</${tag}>`;
    });

  // ── 8. Timestamps → clickable pills ──
  html = html.replace(/\[(\d{1,2}:\d{2}(?::\d{2})?)\]/g,
    '<button class="ts-pill inline-ts" onclick="seekTo(tsToSec(\'$1\'))">▶ $1</button>'
  );

  // ── 9. Newlines → <br> (but NOT inside block elements) ──
  html = html.replace(/\n/g, "<br>");

  // ── 10. Clean up <br> injected inside block tags ──
  html = html
    .replace(/<br>\s*(<\/?(?:table|thead|tbody|tr|th|td|ul|ol|li|h[1-4]|pre|div|hr|blockquote)[^>]*>)/gi, "$1")
    .replace(/(<\/?(?:table|thead|tbody|tr|th|td|ul|ol|li|h[1-4]|pre|div|hr|blockquote)[^>]*>)\s*<br>/gi, "$1");

  // ── 11. Restore stashed LaTeX math as KaTeX-rendered spans ─────────────────
  html = html.replace(/%%MATH_(\d+)%%/g, (_, idx) => {
    const { raw, displayMode } = mathStash[parseInt(idx)];
    try {
      if (typeof katex !== "undefined") {
        return katex.renderToString(raw, {
          displayMode,
          throwOnError: false,
          output: "html",
        });
      }
    } catch (e) {
      console.warn("[KaTeX] Render error:", e.message, raw);
    }
    // Fallback: show raw LaTeX wrapped in a styled span
    return displayMode
      ? `<span class="math-display">\\[${raw}\\]</span>`
      : `<span class="math-inline">\\(${raw}\\)</span>`;
  });

  return html;

}


// ── Phase 2: Practice Quiz ────────────────────────────────────────────────────

function setQuizNum(num) {
  currentQuizNum = num;
  document.querySelectorAll("#quiz-num-toggle .mode-pill").forEach(btn => {
    btn.classList.toggle("active", parseInt(btn.dataset.num) === num);
  });
}

async function generateQuiz() {
  if (!currentVideoId) {
    showToast("Please load a video first.", "error");
    return;
  }

  // Reset state
  userQuizAnswers = {};
  currentQuizId = null;
  currentQuizQuestions = [];

  // Show loading
  showElement(document.getElementById("quiz-loading"));
  hideElement(document.getElementById("quiz-empty"));
  hideElement(document.getElementById("quiz-questions-area"));
  hideElement(document.getElementById("quiz-results-area"));
  document.getElementById("generate-quiz-btn").disabled = true;

  try {
    const resp = await fetch("/api/quiz/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: String(currentVideoId), num_questions: currentQuizNum }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Quiz generation failed.");
    }

    const data = await resp.json();
    currentQuizId = data.quiz_id;
    currentQuizQuestions = data.questions;

    renderQuizQuestions(data.questions);
    showToast(`Generated ${data.total}-question quiz!`, "info");

  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    showElement(document.getElementById("quiz-empty"));
  } finally {
    hideElement(document.getElementById("quiz-loading"));
    document.getElementById("generate-quiz-btn").disabled = false;
  }
}

function renderQuizQuestions(questions) {
  const container = document.getElementById("quiz-questions-list");
  container.innerHTML = "";

  questions.forEach((q, qi) => {
    const card = document.createElement("div");
    card.className = "quiz-question-card";
    card.id = `quiz-q-${q.id}`;

    const optionsHtml = q.options.map((opt, idx) => `
      <label class="quiz-option-label" id="opt-${q.id}-${idx}">
        <input
          type="radio"
          name="quiz_q_${q.id}"
          value="${idx}"
          onchange="selectQuizAnswer(${q.id}, ${idx})"
          class="quiz-radio"
        />
        <span class="option-letter">${String.fromCharCode(65 + idx)}</span>
        <span class="option-text">${escapeHtml(opt)}</span>
      </label>
    `).join("");

    card.innerHTML = `
      <div class="quiz-question-header">
        <span class="quiz-q-num">Q${qi + 1}</span>
        <span class="quiz-q-text">${escapeHtml(q.question)}</span>
      </div>
      <div class="quiz-options-group" id="opts-${q.id}">
        ${optionsHtml}
      </div>
    `;

    container.appendChild(card);
  });

  updateAnsweredCount();
  showElement(document.getElementById("quiz-questions-area"));
  hideElement(document.getElementById("quiz-empty"));
}

function selectQuizAnswer(questionId, selectedIndex) {
  userQuizAnswers[String(questionId)] = selectedIndex;
  updateAnsweredCount();

  // Visual highlight of selected option
  const group = document.getElementById(`opts-${questionId}`);
  if (group) {
    group.querySelectorAll(".quiz-option-label").forEach(label => {
      label.classList.remove("selected");
    });
    const selectedLabel = document.getElementById(`opt-${questionId}-${selectedIndex}`);
    if (selectedLabel) selectedLabel.classList.add("selected");
  }
}

function updateAnsweredCount() {
  const total = currentQuizQuestions.length;
  const answered = Object.keys(userQuizAnswers).length;
  const counter = document.getElementById("quiz-answered-count");
  if (counter) counter.textContent = `${answered} / ${total} answered`;
}

async function submitQuiz() {
  if (!currentQuizId) return;

  const total = currentQuizQuestions.length;
  const answered = Object.keys(userQuizAnswers).length;
  if (answered < total) {
    showToast(`Please answer all ${total} questions before submitting.`, "error");
    return;
  }

  document.getElementById("quiz-submit-btn").disabled = true;

  try {
    const resp = await fetch("/api/quiz/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        quiz_id: currentQuizId,
        video_id: String(currentVideoId),
        answers: userQuizAnswers,
      }),
    });

    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || "Submission failed.");
    }

    const results = await resp.json();
    renderQuizResults(results);

  } catch (err) {
    showToast(`Error: ${err.message}`, "error");
    document.getElementById("quiz-submit-btn").disabled = false;
  }
}

function renderQuizResults(results) {
  const { percentage, correct_count, total, per_question } = results;

  // Animate score ring
  const ring = document.getElementById("score-ring-fill");
  const circumference = 314;
  const offset = circumference - (circumference * percentage / 100);
  ring.style.strokeDashoffset = offset;
  ring.style.stroke = percentage >= 70 ? "#22C55E" : percentage >= 40 ? "#F59E0B" : "#EF4444";

  document.getElementById("score-percent-display").textContent = `${percentage}%`;
  document.getElementById("score-correct").textContent = correct_count;
  document.getElementById("score-wrong").textContent = total - correct_count;
  document.getElementById("score-total").textContent = total;

  // Per-question breakdown
  const list = document.getElementById("quiz-results-list");
  list.innerHTML = "";

  per_question.forEach((pq, idx) => {
    const item = document.createElement("div");
    item.className = `quiz-result-item ${pq.is_correct ? "result-correct" : "result-wrong"}`;

    const tsButton = pq.timestamp > 0
      ? `<button class="ts-pill inline-ts" onclick="seekTo(${pq.timestamp})">▶ ${formatTime(pq.timestamp)}</button>`
      : "";

    item.innerHTML = `
      <div class="result-header">
        <span class="result-badge">${pq.is_correct ? "✅" : "❌"}</span>
        <span class="result-q-num">Q${idx + 1}</span>
        <span class="result-q-text">${escapeHtml(pq.question)}</span>
        ${tsButton}
      </div>
      <div class="result-options">
        ${pq.options.map((opt, i) => `
          <div class="result-option ${i === pq.correct_index ? "correct-opt" : i === pq.selected_index && !pq.is_correct ? "wrong-opt" : ""}">
            <span class="option-letter">${String.fromCharCode(65 + i)}</span>
            <span>${escapeHtml(opt)}</span>
            ${i === pq.correct_index ? '<span class="opt-tag correct-tag">✓ Correct</span>' : ""}
            ${i === pq.selected_index && !pq.is_correct ? '<span class="opt-tag wrong-tag">✗ Your answer</span>' : ""}
          </div>
        `).join("")}
      </div>
      <div class="result-explanation">
        <span class="exp-icon">💡</span>
        <span>${escapeHtml(pq.explanation)}</span>
      </div>
    `;

    list.appendChild(item);
  });

  hideElement(document.getElementById("quiz-questions-area"));
  showElement(document.getElementById("quiz-results-area"));

  const grade = percentage >= 80 ? "Excellent! 🎉" : percentage >= 60 ? "Good job! 👍" : "Keep studying! 📚";
  showToast(`${grade} You scored ${percentage}% (${correct_count}/${total})`, "info");
}

function retakeQuiz() {
  // Reset everything and go back to generate state
  userQuizAnswers = {};
  currentQuizId = null;
  currentQuizQuestions = [];
  hideElement(document.getElementById("quiz-results-area"));
  hideElement(document.getElementById("quiz-questions-area"));
  showElement(document.getElementById("quiz-empty"));
  document.getElementById("generate-quiz-btn").disabled = false;
}


// ── Theme Switcher (ThetaWave Light / Obsidian Dark) ──────────────────────────
function initTheme() {
  const savedTheme = localStorage.getItem("lecturemind_theme") || "light"; // default to ThetaWave light theme
  applyTheme(savedTheme);
}

function toggleTheme() {
  const isLight = document.body.classList.contains("theme-light");
  const newTheme = isLight ? "obsidian" : "light";
  applyTheme(newTheme);
  localStorage.setItem("lecturemind_theme", newTheme);
  showToast(`Switched to ${newTheme === "light" ? "ThetaWave Light" : "Obsidian Dark"} theme`);
}

function applyTheme(themeName) {
  const iconContainer = document.getElementById("theme-icon-container");
  const label = document.getElementById("theme-toggle-label");

  if (themeName === "light") {
    document.body.classList.remove("theme-obsidian");
    document.body.classList.add("theme-light");
    if (label) label.textContent = "Dark Mode";
    if (iconContainer) {
      iconContainer.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
        </svg>
      `;
    }
  } else {
    document.body.classList.remove("theme-light");
    document.body.classList.add("theme-obsidian");
    if (label) label.textContent = "Light Mode";
    if (iconContainer) {
      iconContainer.innerHTML = `
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="5"></circle>
          <line x1="12" y1="1" x2="12" y2="3"></line>
          <line x1="12" y1="21" x2="12" y2="23"></line>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
          <line x1="1" y1="12" x2="3" y2="12"></line>
          <line x1="21" y1="12" x2="23" y2="12"></line>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
        </svg>
      `;
    }
  }
}

function copyCodeBlock(blockId) {
  const container = document.getElementById(blockId);
  if (!container) return;
  const codeEl = container.querySelector(".code-content");
  if (!codeEl) return;
  const text = codeEl.innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = container.querySelector(".code-copy-btn");
    if (btn) {
      const origHtml = btn.innerHTML;
      btn.innerHTML = `<span>✓ Copied</span>`;
      btn.classList.add("copied");
      setTimeout(() => {
        btn.innerHTML = origHtml;
        btn.classList.remove("copied");
      }, 2000);
    }
  }).catch(() => {
    showToast("Failed to copy code", "error");
  });
}

// Global initialization
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  // Always display the main page by default when entering the website
  switchMainView("onboarding");
  // Initialize Auth state & load user library
  initAuthSession();
  const heroInput = document.getElementById("youtube-url-input");
  if (heroInput) {
    heroInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") handleProcessVideo();
    });
  }
});




