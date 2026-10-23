package usagelog

// Generates the dashboard's Connect usage-export fixtures with Connect's own
// export code. This file is not part of Connect: `just connect-fixtures` (in the
// dashboard repo) adds it to connect/gateway/usagelog with `go test -overlay`,
// inside a connect-dev container, without writing to the Connect checkout.
//
// It lays down usage-log JSONL the way the writer does (Record lines under
// date=<ingest day>/node=<node>/events.jsonl), then asks ExportStore for the
// window at two watermarks and copies each Parquet file to DASHBOARD_FIXTURE_OUT:
//
//   connect_export.parquet        as_of 2026-10-01: every record below
//   connect_export_early.parquet  as_of 2026-09-20T12:00: what a load at that
//                                 time saw, for incremental-merge tests
//
// Each record's response_id names the case it covers; tests/test_connect_export.py
// relies on them.

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func ptr[T any](v T) *T { return &v }

func at(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return t
}

// fixtureRecord is a successful Workbench call by alice; each case edits it.
func fixtureRecord(id, start, ingest string, edit func(*Record)) Record {
	r := Record{
		StartTime:             at(start),
		IngestTime:            at(ingest),
		NodeName:              "connect-01",
		DurationMs:            1500,
		Provider:              "anthropic",
		Operation:             "chat",
		IntegrationGUID:       "integration-anthropic",
		IntegrationType:       "anthropic",
		AuthType:              "Viewer",
		AuthenticatedUserGUID: ptr("guid-alice"),
		AuthenticatedUsername: ptr("alice"),
		AuthenticatedEmail:    ptr("alice@example.com"),
		ConversationID:        ptr("conv-alice-1"),
		UserAgent:             ptr("PositAssistant/1.2.3 posit_workbench_positron/2026.01.0"),
		RequestMetadata: map[string]string{
			"team": "data-science", "workbench-session-id": "d1ac60de",
		},
		ServerAddress:            "api.anthropic.com",
		ServerPort:               443,
		StatusCode:               200,
		RequestModel:             "claude-sonnet-4-5",
		ResponseModel:            "claude-sonnet-4-5-20250929",
		ResponseID:               id,
		RequestMaxTokens:         8192,
		RequestTemperature:       ptr(1.0),
		FinishReasons:            []string{"end_turn"},
		InputTokens:              12000,
		OutputTokens:             800,
		CacheCreationInputTokens: 2000,
		CacheReadInputTokens:     9000,
	}
	if edit != nil {
		edit(&r)
	}
	return r
}

func fixtureRecords() []Record {
	return []Record{
		fixtureRecord("workbench-assistant", "2026-09-10T14:00:00Z", "2026-09-10T14:00:02Z", nil),
		fixtureRecord("local-assistant", "2026-09-10T15:00:00Z", "2026-09-10T15:00:02Z", func(r *Record) {
			r.UserAgent = ptr("PositAssistant/1.2.3 posit_rstudio/2026.01.0")
			r.RequestMetadata = nil
		}),
		fixtureRecord("claude-code", "2026-09-11T09:00:00Z", "2026-09-11T09:00:03Z", func(r *Record) {
			r.UserAgent = ptr("claude-cli/2.0.14 (external, cli)")
			r.RequestMetadata = map[string]string{"team": "platform", "cost-center": "cc-42"}
			r.ConversationID = ptr("conv-alice-2")
		}),
		// Content acting for its viewer, through a Bedrock US inference profile.
		fixtureRecord("viewer-content", "2026-09-12T10:00:00Z", "2026-09-12T10:00:04Z", func(r *Record) {
			r.NodeName = "connect-02"
			r.Provider = "aws.bedrock"
			r.IntegrationGUID = "integration-bedrock"
			r.IntegrationType = "aws"
			r.ServerAddress = "bedrock-runtime.us-east-1.amazonaws.com"
			r.RequestModel = "us.anthropic.claude-opus-4-1-20250805-v1:0"
			r.ResponseModel = "us.anthropic.claude-opus-4-1-20250805-v1:0"
			r.ContentGUID = ptr("content-churn")
			r.ContentID = ptr(int64(7))
			r.JobKey = ptr("job-1")
			r.AuthenticatedUserGUID, r.AuthenticatedUsername, r.AuthenticatedEmail = ptr("guid-vic"), ptr("vic"), ptr("vic@example.com")
			r.ViewerUserGUID, r.ViewerUsername, r.ViewerEmail = ptr("guid-vic"), ptr("vic"), ptr("vic@example.com")
			r.ContentOwnerGUID, r.ContentOwnerUsername, r.ContentOwnerEmail = ptr("guid-olga"), ptr("olga"), ptr("olga@example.com")
			r.UserAgent = ptr("r-ellmer/0.3.2")
			r.RequestMetadata = nil
			r.ConversationID = nil
		}),
		// Content on its owner's service account: only the owner is known.
		fixtureRecord("service-content", "2026-09-12T11:00:00Z", "2026-09-12T11:00:01Z", func(r *Record) {
			r.AuthType = "Service Account"
			r.ContentGUID = ptr("content-etl")
			r.ContentID = ptr(int64(8))
			r.JobKey = ptr("job-2")
			r.AuthenticatedUserGUID, r.AuthenticatedUsername, r.AuthenticatedEmail = nil, nil, nil
			r.ContentOwnerGUID, r.ContentOwnerUsername, r.ContentOwnerEmail = ptr("guid-olga"), ptr("olga"), ptr("olga@example.com")
			r.UserAgent = ptr("python-httpx/0.27.2")
			r.RequestMetadata = nil
			r.ConversationID = nil
			r.ResponseModel = "claude-haiku-4-5-20251001"
			r.RequestModel = "claude-haiku-4-5"
		}),
		// A refused call: empty response model, no output.
		fixtureRecord("failed-call", "2026-09-13T08:00:00Z", "2026-09-13T08:00:00Z", func(r *Record) {
			r.StatusCode = 429
			r.ResponseModel = ""
			r.FinishReasons = nil
			r.OutputTokens = 0
			r.CacheCreationInputTokens, r.CacheReadInputTokens = 0, 0
			r.InputTokens = 0
		}),
		// The name lookup was off: GUID only.
		fixtureRecord("no-name", "2026-09-14T08:00:00Z", "2026-09-14T08:00:01Z", func(r *Record) {
			r.AuthenticatedUsername, r.AuthenticatedEmail = nil, nil
			r.UserAgent = nil
			r.RequestMetadata = nil
		}),
		// Ingested just after the early export's watermark minus the refetch margin.
		fixtureRecord("near-cutoff", "2026-09-20T11:55:00Z", "2026-09-20T11:55:05Z", nil),
		// Started early in the month, ingested only after the early export.
		fixtureRecord("late-ingest", "2026-09-05T09:00:00Z", "2026-09-21T09:00:00Z", nil),
		fixtureRecord("after-early", "2026-09-22T15:00:00Z", "2026-09-22T15:00:01Z", func(r *Record) {
			r.NodeName = "connect-02"
		}),
		// Started before the window; its partition is inside it.
		fixtureRecord("before-window", "2026-08-31T23:59:59Z", "2026-09-01T00:00:01Z", nil),
	}
}

// oldRecordLine is a record written before user_agent, request_metadata and
// conversation_id existed: those keys are absent from the line.
const oldRecordLine = `{"start_time":"2026-09-15T08:00:00Z","ingest_time":"2026-09-15T08:00:01Z",` +
	`"node_name":"connect-01","duration_ms":900,"provider":"anthropic","operation":"chat",` +
	`"integration_guid":"integration-anthropic","integration_type":"anthropic",` +
	`"integration_auth_type":"Viewer","content_guid":null,"content_id":null,"job_key":null,` +
	`"authenticated_user_guid":"guid-alice","authenticated_username":"alice",` +
	`"authenticated_email":"alice@example.com","content_owner_guid":null,` +
	`"content_owner_username":null,"content_owner_email":null,"viewer_user_guid":null,` +
	`"viewer_username":null,"viewer_email":null,"server_address":"api.anthropic.com",` +
	`"server_port":443,"status_code":200,"request_model":"claude-sonnet-4-5",` +
	`"response_model":"claude-sonnet-4-5-20250929","response_id":"old-record",` +
	`"finish_reasons":["end_turn"],"input_tokens":500,"output_tokens":50,` +
	`"cache_creation_input_tokens":0,"cache_read_input_tokens":0}`

func TestDashboardFixture(t *testing.T) {
	out := os.Getenv("DASHBOARD_FIXTURE_OUT")
	if out == "" {
		t.Skip("DASHBOARD_FIXTURE_OUT is not set")
	}
	dir := t.TempDir()

	// One events.jsonl per (ingest day, node), as the writer partitions them.
	files := map[string][]string{}
	for _, r := range fixtureRecords() {
		line, err := json.Marshal(r)
		if err != nil {
			t.Fatal(err)
		}
		part := filepath.Join(dir, "date="+partitionDate(r.IngestTime), "node="+r.NodeName)
		files[part] = append(files[part], string(line))
	}
	oldPart := filepath.Join(dir, "date=2026-09-15", "node=connect-01")
	files[oldPart] = append(files[oldPart], oldRecordLine)
	for part, lines := range files {
		if err := os.MkdirAll(part, 0700); err != nil {
			t.Fatal(err)
		}
		body := strings.Join(lines, "\n") + "\n"
		if err := os.WriteFile(filepath.Join(part, "events.jsonl"), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}

	now := at("2026-10-02T00:00:00Z")
	store := NewExportStore(ExportOptions{
		Dir: dir, NodeName: "connect-01", Now: func() time.Time { return now },
	})
	window := Window{From: at("2026-09-01T00:00:00Z"), To: at("2026-10-01T00:00:00Z")}
	for name, asOf := range map[string]string{
		"connect_export.parquet":       "2026-10-01T00:00:00Z",
		"connect_export_early.parquet": "2026-09-20T12:00:00Z",
	} {
		exp, err := store.Get(context.Background(), Spec{Window: window, asOf: at(asOf)})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		body, err := os.ReadFile(exp.Path)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(out, name), body, 0644); err != nil {
			t.Fatal(err)
		}
	}
}
