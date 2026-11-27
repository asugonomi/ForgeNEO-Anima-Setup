using System;
using System.Diagnostics;
using System.IO;
using System.Text;

internal static class Launcher
{
    public static int Main()
    {
        Console.OutputEncoding = new UTF8Encoding(false);
        string root = AppDomain.CurrentDomain.BaseDirectory;
        bool update = Path.GetFileNameWithoutExtension(Environment.GetCommandLineArgs()[0])
                          .EndsWith("-Update", StringComparison.OrdinalIgnoreCase);
        Console.Title = update ? "ForgeNEO 更新" : "ForgeNEO 起動ログ";
        try
        {
            var start = new ProcessStartInfo(Path.Combine(root, "python", "python.exe"),
                "-E -s -X utf8 -u manager.py " + (update ? "update" : "launch"));
            start.WorkingDirectory = root;
            start.UseShellExecute = false;
            using (var child = Process.Start(start))
            {
                Console.CancelKeyPress += delegate(object sender, ConsoleCancelEventArgs e) { e.Cancel = true; };
                child.WaitForExit();
                if (child.ExitCode != 0 && !Console.IsInputRedirected)
                {
                    Console.WriteLine("Enter キーで閉じます。");
                    Console.ReadLine();
                }
                return child.ExitCode;
            }
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.Message);
            if (!Console.IsInputRedirected) Console.ReadLine();
            return 1;
        }
    }
}
