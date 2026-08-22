const roomsElement = document.querySelector('#rooms');
const summaryElement = document.querySelector('#room-summary');
const backendElement = document.querySelector('#backend');
const newRoomButton = document.querySelector('#new-room');
const joinForm = document.querySelector('#join-form');
const roomCodeInput = document.querySelector('#room-code');
const joinError = document.querySelector('#join-error');

function roomIdFromInput(value) {
  const trimmed = value.trim();
  if (!trimmed) return '';
  let candidate = trimmed;
  try {
    const url = new URL(trimmed, location.origin);
    candidate = url.searchParams.get('session') || trimmed;
  } catch {
    // Treat non-URL text as a room code.
  }
  return /^[A-Za-z0-9_-]{1,64}$/.test(candidate) ? candidate : '';
}

function enterRoom(sessionId) {
  location.href = `/?session=${encodeURIComponent(sessionId)}`;
}

function stateFor(room) {
  if (room.job_status === 'running' || room.job_status === 'queued') return ['Reconstructing', ''];
  if (room.processing_videos) return ['Processing video', ''];
  if (room.update_pending) return ['Update pending', ''];
  if (room.viewer_ready) return ['3D room ready', 'ready'];
  if (room.frame_count) return ['Frames received', ''];
  return ['Waiting for scans', ''];
}

function relativeTime(value) {
  const seconds = Math.max(0, (Date.now() - new Date(value).getTime()) / 1000);
  if (seconds < 60) return 'now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

function roomCard(room) {
  const article = document.createElement('article');
  article.className = 'room';
  const thumbnail = document.createElement('div');
  thumbnail.className = 'thumb';
  if (room.thumbnail_url) {
    const image = document.createElement('img');
    image.src = room.thumbnail_url;
    image.alt = '';
    image.loading = 'lazy';
    image.addEventListener('error', () => image.remove(), { once: true });
    thumbnail.append(image);
  }
  const [stateText, stateClass] = stateFor(room);
  const state = document.createElement('div');
  state.className = `state ${stateClass}`.trim();
  state.textContent = stateText;
  thumbnail.append(state);

  const body = document.createElement('div');
  body.className = 'room-body';
  const titleRow = document.createElement('div');
  titleRow.className = 'room-title';
  const title = document.createElement('h3');
  title.textContent = room.title;
  const updated = document.createElement('time');
  updated.dateTime = room.updated_at;
  updated.textContent = relativeTime(room.updated_at);
  titleRow.append(title, updated);
  const code = document.createElement('div');
  code.className = 'code';
  code.textContent = room.session_id;
  const stats = document.createElement('div');
  stats.className = 'stats';
  stats.textContent = `${room.client_count} contributor${room.client_count === 1 ? '' : 's'}  ·  ${room.frame_count} frames`;
  const actions = document.createElement('div');
  actions.className = 'room-actions';
  const join = document.createElement('a');
  join.className = 'button';
  join.href = room.capture_url;
  join.textContent = 'Join scan';
  const view = document.createElement('a');
  view.className = `button secondary${room.viewer_ready ? '' : ' disabled'}`;
  view.href = room.viewer_url;
  view.textContent = 'View room';
  if (!room.viewer_ready) view.setAttribute('aria-disabled', 'true');
  actions.append(join, view);
  body.append(titleRow, code, stats, actions);
  article.append(thumbnail, body);
  return article;
}

async function loadRooms() {
  try {
    const response = await fetch('/api/sessions', { cache: 'no-store' });
    if (!response.ok) throw new Error('Could not load rooms');
    const payload = await response.json();
    backendElement.textContent = `${payload.backend} box online`;
    summaryElement.textContent = `${payload.sessions.length} room${payload.sessions.length === 1 ? '' : 's'} on this box`;
    roomsElement.replaceChildren();
    if (!payload.sessions.length) {
      const empty = document.createElement('div');
      empty.id = 'empty';
      empty.textContent = 'No rooms yet. Start the first scan.';
      roomsElement.append(empty);
      return;
    }
    roomsElement.append(...payload.sessions.map(roomCard));
  } catch (error) {
    backendElement.textContent = 'Box unavailable';
    summaryElement.textContent = error.message;
  }
}

newRoomButton.addEventListener('click', async () => {
  newRoomButton.disabled = true;
  newRoomButton.textContent = 'Creating room…';
  try {
    const response = await fetch('/api/sessions', { method: 'POST' });
    if (!response.ok) throw new Error('Could not create a room');
    const room = await response.json();
    enterRoom(room.session_id);
  } catch (error) {
    joinError.textContent = error.message;
    newRoomButton.disabled = false;
    newRoomButton.textContent = '＋ New room scan';
  }
});

joinForm.addEventListener('submit', (event) => {
  event.preventDefault();
  const sessionId = roomIdFromInput(roomCodeInput.value);
  if (!sessionId) {
    joinError.textContent = 'Enter a valid room code or ROOMSCAN link.';
    return;
  }
  joinError.textContent = '';
  enterRoom(sessionId);
});

await loadRooms();
setInterval(loadRooms, 5000);
