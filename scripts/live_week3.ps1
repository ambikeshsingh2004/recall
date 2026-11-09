# Live check of week-3 features through the running API (forget, correction, alias, time filter, search).
$ErrorActionPreference = "Stop"
$base = "http://localhost:8000"
$user = "live-" + (Get-Date -Format "HHmmss")

function Chat($thread, $msg) {
  $body = @{ user_id = $user; thread_id = $thread; message = $msg; mode = "ours" } | ConvertTo-Json
  $r = Invoke-RestMethod "$base/chat" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 180
  "[$thread] you: $msg"
  "[$thread] bot ($($r.latency_ms) ms): $($r.answer)`n"
}
function EndThread($thread) { Invoke-RestMethod "$base/threads/$thread/end?user_id=$user" -Method Post -TimeoutSec 180 | Out-Null }
function WaitStoring { Start-Sleep -Seconds 12 }

Chat "t1" "Hi, I'm Sam, a designer at Pixelcraft. My partner Leah and I are going to Japan in December. My phone number is 98765 43210."
WaitStoring; EndThread "t1"
Chat "t2" "Please forget my phone number. Also I said Japan is in December, that's wrong, it's in March."
WaitStoring; EndThread "t2"

"--- memories now (current + outdated)"
$mems = Invoke-RestMethod "$base/memories?user_id=$user&include_old=true"
foreach ($m in $mems) {
  $state = if ($m.valid_to) { "OUTDATED" } else { "current " }
  "  $state [$($m.type)] $($m.text)"
}
"--- learned user name"
(Invoke-RestMethod "$base/profile?user_id=$user").user_id | Out-Null
docker compose exec -T db psql -U recall -d recall -tA -c "SELECT user_name FROM profiles WHERE user_id = '$user'"
"--- entities and aliases"
docker compose exec -T db psql -U recall -d recall -tA -c "SELECT name || '  aliases=' || array_to_string(aliases, ', ') FROM entities WHERE user_id = '$user'"

Chat "t3" "What's my phone number, and when is the trip with my partner?"

"--- /memories/search for a multi-part question"
$s = Invoke-RestMethod "$base/memories/search?user_id=$user&q=$([uri]::EscapeDataString('Who is my partner and where do I work?'))" -TimeoutSec 120
"analyzed=$($s.analyzed) latency=$($s.latency_ms) ms"
foreach ($sq in $s.subqueries) {
  "  sub-query: $($sq.search_query) | entities: $($sq.entities -join ', ')"
  foreach ($n in $sq.notes) { "    score=$($n.score) ranks=$($n.ranks | ConvertTo-Json -Compress) $($n.text)" }
}
