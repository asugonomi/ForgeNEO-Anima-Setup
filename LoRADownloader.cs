using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace ForgeNeoLora
{
    internal sealed class Downloader : Form
    {
        readonly TextBox key = new TextBox { UseSystemPasswordChar = true };
        readonly TextBox folder = new TextBox();
        readonly Button browse = new Button { Text = "参照…" };
        readonly Button fetch = new Button { Text = "投稿一覧を取得" };
        readonly Button download = new Button { Text = "選んだ分類をダウンロード" };
        readonly Button cancel = new Button { Text = "中止", Enabled = false };
        readonly CheckBox older = new CheckBox { Text = "旧バージョンも含める", AutoSize = true };
        readonly CheckBox[] groups = new CheckBox[Catalog.Groups.Length];
        readonly Label summary = new Label { AutoSize = true };
        readonly TextBox log = new TextBox { Multiline = true, ReadOnly = true,
            ScrollBars = ScrollBars.Vertical, Dock = DockStyle.Fill };
        List<LoraFile> catalog;
        CancellationTokenSource running;
        bool closing;

        [STAThread]
        static void Main()
        {
            ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12;
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Downloader());
        }

        internal Downloader()
        {
            Text = "ForgeNEO LoRA ダウンローダー";
            Font = new Font("Yu Gothic UI", 10);
            AutoScaleMode = AutoScaleMode.Dpi;
            ClientSize = new Size(690, 530);
            MinimumSize = new Size(640, 530);
            var grid = new TableLayoutPanel { Dock = DockStyle.Fill, Padding = new Padding(12),
                ColumnCount = 3, RowCount = 9 };
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 100));
            grid.ColumnStyles.Add(new ColumnStyle(SizeType.AutoSize));
            for (int i = 0; i < 8; i++) grid.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            grid.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
            Controls.Add(grid);
            AddWide(grid, new Label { Text = "asugonomi が Civitai に投稿した LoRA", AutoSize = true }, 0);
            grid.Controls.Add(new Label { Text = "Civitai APIキー", AutoSize = true }, 0, 1);
            key.Dock = DockStyle.Fill;
            grid.Controls.Add(key, 1, 1);
            fetch.AutoSize = true;
            grid.Controls.Add(fetch, 2, 1);
            grid.Controls.Add(new Label { Text = "LoRA 保存先", AutoSize = true }, 0, 2);
            folder.Dock = DockStyle.Fill;
            var parent = AppDomain.CurrentDomain.BaseDirectory;
            if (Directory.Exists(Path.Combine(parent, "ForgeNEO-Anima", "forge")))
                parent = Path.Combine(parent, "ForgeNEO-Anima");
            folder.Text = Directory.Exists(Path.Combine(parent, "forge")) ?
                Path.Combine(parent, "forge", "models", "Lora") : Path.Combine(parent, "Lora");
            grid.Controls.Add(folder, 1, 2);
            grid.Controls.Add(browse, 2, 2);
            AddWide(grid, older, 3);
            var choices = new TableLayoutPanel { ColumnCount = 2, RowCount = 3,
                AutoSize = true, Dock = DockStyle.Top };
            for (int i = 0; i < groups.Length; i++)
            {
                var choice = new CheckBox { AutoSize = true, Text = Catalog.Groups[i] };
                groups[i] = choice;
                choices.Controls.Add(choice, i / 3, i % 3);
                choice.CheckedChanged += delegate { RefreshSelection(); };
            }
            AddWide(grid, choices, 4);
            AddWide(grid, summary, 5);
            var actions = new FlowLayoutPanel { AutoSize = true, Dock = DockStyle.Top };
            download.AutoSize = true;
            actions.Controls.Add(download);
            actions.Controls.Add(cancel);
            AddWide(grid, actions, 6);
            AddWide(grid, new Label { Text = "処理ログ", AutoSize = true }, 7);
            AddWide(grid, log, 8);
            browse.Click += delegate {
                using (var dialog = new FolderBrowserDialog { Description = "LoRAの保存先",
                    SelectedPath = folder.Text })
                    if (dialog.ShowDialog(this) == DialogResult.OK) folder.Text = dialog.SelectedPath;
            };
            key.TextChanged += delegate { catalog = null; RefreshSelection(); };
            older.CheckedChanged += delegate { RefreshSelection(); };
            fetch.Click += async delegate { await Run(false); };
            download.Click += async delegate { await Run(true); };
            cancel.Click += delegate { if (running != null) running.Cancel(); };
            FormClosing += delegate(object sender, FormClosingEventArgs e) {
                if (running != null) { closing = true; running.Cancel(); e.Cancel = true; }
            };
            RefreshSelection();
        }

        static void AddWide(TableLayoutPanel grid, Control child, int row)
        {
            grid.Controls.Add(child, 0, row);
            grid.SetColumnSpan(child, 3);
        }

        List<LoraFile> Selected(List<LoraFile> versions)
        {
            var enabled = new HashSet<string>(Catalog.Groups.Where((g,i) => groups[i].Checked));
            return versions.Where(f => enabled.Contains(f.Group)).ToList();
        }

        static string SizeText(long bytes)
        {
            return (bytes / 1073741824.0).ToString("0.00", CultureInfo.InvariantCulture) + " GiB";
        }

        void RefreshSelection()
        {
            var versions = Catalog.SelectVersions(catalog ?? new List<LoraFile>(), older.Checked);
            for (int i = 0; i < groups.Length; i++)
            {
                string group = Catalog.Groups[i];
                groups[i].Text = group + (catalog == null ? "" : "  (" + versions.Count(f => f.Group == group) + " 件)");
                groups[i].Enabled = running == null && catalog != null;
            }
            var selected = Selected(versions);
            summary.Text = catalog == null ? "APIキーを入力し、投稿一覧を取得してください。" :
                "選択中　" + selected.Count + " 件　合計約 " + SizeText(selected.Sum(f => f.Bytes));
            download.Enabled = running == null && selected.Count > 0;
            fetch.Enabled = running == null && !string.IsNullOrWhiteSpace(key.Text);
        }

        async Task Run(bool save)
        {
            if (running != null) return;
            var source = new CancellationTokenSource();
            running = source;
            var token = source.Token;
            string secret = key.Text.Trim();
            key.Enabled = folder.Enabled = browse.Enabled = older.Enabled = false;
            cancel.Enabled = true;
            RefreshSelection();
            IProgress<string> progress = new Progress<string>(s => { if (!IsDisposed) log.AppendText(s + Environment.NewLine); });
            try
            {
                if (!save)
                {
                    catalog = null;
                    catalog = await Task.Run(() => Catalog.ListFiles(secret, token, progress.Report), token);
                    log.AppendText("投稿一覧を取得しました。" + Environment.NewLine);
                }
                else
                {
                    if (string.IsNullOrWhiteSpace(folder.Text) || !Path.IsPathRooted(folder.Text))
                        throw new IOException("保存先には絶対パスを指定してください。");
                    string root = Path.GetFullPath(folder.Text.Trim());
                    var selected = Selected(Catalog.SelectVersions(catalog, older.Checked));
                    await Task.Run(() => Download(selected, root, secret, token, progress.Report), token);
                }
            }
            catch (Exception ex)
            {
                string detail = token.IsCancellationRequested ? "中止しました。" : ex.Message.Replace(secret, "****");
                log.AppendText(detail + Environment.NewLine);
            }
            finally
            {
                running = null;
                source.Dispose();
                key.Enabled = folder.Enabled = browse.Enabled = older.Enabled = true;
                cancel.Enabled = false;
                RefreshSelection();
                if (closing) Close();
            }
        }

        internal static void Download(List<LoraFile> selected, string root, string secret,
            CancellationToken token, Action<string> report)
        {
            Directory.CreateDirectory(root);
            var pending = new List<LoraFile>();
            int existing = 0, failed = 0, saved = 0;
            report("保存済みファイルを確認中…");
            foreach (var file in selected)
            {
                token.ThrowIfCancellationRequested();
                string path = file.Target(root);
                if (!File.Exists(path)) pending.Add(file);
                else if (Catalog.Matches(path, file.Sha, token)) existing++;
                else { failed++; report("同名ファイルの内容が異なります。確認してください　" + path); }
            }
            long needed = pending.Sum(f => f.Bytes);
            string volume = Path.GetPathRoot(Path.GetFullPath(root));
            if (!volume.StartsWith(@"\\") && needed > new DriveInfo(volume).AvailableFreeSpace)
                throw new IOException("空き容量が不足しています。未取得分は約 " + SizeText(needed) + " です。");
            foreach (var file in pending)
            {
                token.ThrowIfCancellationRequested();
                report("取得中　" + file.Group + "　" + file.FileName);
                try
                {
                    Catalog.ReadUrl(file.Download, secret, token,
                        stream => Catalog.SaveVerified(stream, file.Target(root), file.Sha, token));
                    saved++;
                }
                catch (Exception ex)
                {
                    token.ThrowIfCancellationRequested();
                    failed++;
                    report("失敗　" + file.FileName + "　" + ex.Message);
                    if ((ex.HResult & 65535) == 112 || (ex.HResult & 65535) == 39 ||
                        ex.Message.StartsWith("HTTP 401") || ex.Message.StartsWith("HTTP 429")) break;
                }
            }
            report("取得 " + saved + " 件　保存済み " + existing + " 件　失敗 " + failed +
                " 件　未処理 " + (selected.Count - saved - existing - failed) + " 件");
        }
    }
}
