using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Security.Cryptography;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;

namespace ForgeNeoLora
{
    internal sealed class LoraFile
    {
        public string Family, Category, Name, FileName, Sha;
        public long ModelId, VersionId, FileId, Bytes;
        public DateTimeOffset Published;
        public Uri Download;
        public string Group { get { return Family + " / " + Category; } }
        public string Target(string root)
        {
            return Path.Combine(root, Family, Category, ModelId + "-" + VersionId, FileName);
        }
    }

    internal static class Catalog
    {
        internal const string Author = "asugonomi";
        internal static readonly string[] Groups = (from family in new[] {"Anima", "Illustrious"}
            from category in new[] {"ウマ娘", "その他キャラ", "コンセプト"} select family + " / " + category).ToArray();
        static readonly JavaScriptSerializer Json = new JavaScriptSerializer { MaxJsonLength = 32 * 1024 * 1024 };
        static readonly Dictionary<string,object> Empty = new Dictionary<string,object>();
        internal static Dictionary<string,object> Obj(object value) { return value as Dictionary<string,object> ?? Empty; }
        internal static object Get(Dictionary<string,object> obj, string key) { object v; return obj.TryGetValue(key,out v) ? v : null; }
        static string Text(Dictionary<string,object> obj,string key) { return Convert.ToString(Get(obj,key),CultureInfo.InvariantCulture); }
        static long Id(Dictionary<string,object> obj,string key) { long n; return long.TryParse(Text(obj,key),out n) ? n : 0; }
        static IEnumerable<object> Items(Dictionary<string,object> obj,string key)
        {
            var values = Get(obj,key) as IList;
            return values == null ? Enumerable.Empty<object>() : values.Cast<object>();
        }
        internal static bool Api(Uri url)
        {
            return url != null && url.IsAbsoluteUri && url.Scheme=="https" && url.IsDefaultPort &&
                url.Host.Equals("civitai.com",StringComparison.OrdinalIgnoreCase) &&
                url.UserInfo.Length==0 && url.AbsolutePath.StartsWith("/api/",StringComparison.Ordinal);
        }
        static string Classify(Dictionary<string,object> model)
        {
            // These publications have missing character tags or describe a pose, style or uniform.
            switch (Id(model,"id"))
            {
                case 1607207: return "その他キャラ";
                case 1564776: case 1590838: case 1344628: return "コンセプト";
            }
            var tags = new HashSet<string>(Items(model,"tags").Select(x=>Convert.ToString(x).Trim()),StringComparer.OrdinalIgnoreCase);
            string label = Text(model,"name")+" "+string.Join(" ",tags);
            if (Regex.IsMatch(label,@"uma[\s_-]*musume|ウマ娘",RegexOptions.IgnoreCase)) return "ウマ娘";
            if (tags.Contains("character") || tags.Contains("characters") || tags.Contains("キャラクター")) return "その他キャラ";
            return "コンセプト";
        }
        internal static string SafeName(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw) || raw!=Path.GetFileName(raw) ||
                raw.IndexOfAny(Path.GetInvalidFileNameChars())>=0 ||
                !raw.EndsWith(".safetensors",StringComparison.OrdinalIgnoreCase) ||
                Regex.IsMatch(raw,@"^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)",RegexOptions.IgnoreCase)) return null;
            return raw;
        }
        internal static List<LoraFile> ParsePage(string json, out string next)
        {
            var page=Json.Deserialize<Dictionary<string,object>>(json);
            if (!(Get(page,"items") is IList)) throw new InvalidDataException("モデル一覧を読み取れません。");
            next=Text(Obj(Get(page,"metadata")),"nextPage");
            var result=new List<LoraFile>();
            foreach(var model in Items(page,"items").Select(Obj))
            {
                if (!Author.Equals(Text(Obj(Get(model,"creator")),"username"),StringComparison.OrdinalIgnoreCase)
                    || !Text(model,"type").Equals("LORA",StringComparison.OrdinalIgnoreCase)) continue;
                string category=Classify(model);
                foreach(var ver in Items(model,"modelVersions").Select(Obj))
                {
                    string family=Text(ver,"baseModel");
                    if(family!="Anima" && family!="Illustrious") continue;
                    var files=Items(ver,"files").Select(Obj)
                        .Where(f=>Text(f,"type")=="Model" && SafeName(Text(f,"name"))!=null)
                        .OrderByDescending(f=>object.Equals(Get(f,"primary"),true));
                    foreach(var f in files)
                    {
                        string sha=Text(Obj(Get(f,"hashes")),"SHA256");
                        Uri address;
                        long mid=Id(model,"id"), vid=Id(ver,"id"), fid=Id(f,"id");
                        if(mid<=0 || vid<=0 || fid<=0 || !Regex.IsMatch(sha,"^[0-9a-fA-F]{64}$") ||
                            !Uri.TryCreate(Text(f,"downloadUrl"),UriKind.Absolute,out address) || !Api(address) ||
                            address.AbsolutePath!="/api/download/models/"+vid) continue;
                        DateTimeOffset published;
                        DateTimeOffset.TryParse(Text(ver,"publishedAt"),CultureInfo.InvariantCulture,DateTimeStyles.AssumeUniversal,out published);
                        result.Add(new LoraFile {Family=family,Category=category,Name=Text(model,"name"),
                            FileName=SafeName(Text(f,"name")),Sha=sha.ToLowerInvariant(),ModelId=mid,VersionId=vid,
                            FileId=fid,Published=published,Download=address,Bytes=(long)Math.Ceiling(
                                Convert.ToDouble(Get(f,"sizeKB"),CultureInfo.InvariantCulture)*1024)});
                        break;
                    }
                }
            }
            return result;
        }
        internal static List<LoraFile> SelectVersions(IEnumerable<LoraFile> files,bool older)
        {
            var distinct=files.GroupBy(f=>f.FileId).Select(g=>g.First());
            if(!older) distinct=distinct.GroupBy(f=>new {f.ModelId,f.Family})
                .Select(g=>g.OrderByDescending(f=>f.Published).ThenByDescending(f=>f.VersionId).First());
            return distinct.OrderBy(f=>f.Group).ThenBy(f=>f.ModelId).ThenByDescending(f=>f.VersionId).ToList();
        }
        internal static void ReadUrl(Uri start,string key,CancellationToken cancel,Action<Stream> read)
        {
            if(!Api(start)) throw new InvalidDataException("Civitai APIのURLを確認できません。");
            Uri url=start;
            bool sendKey=true;
            for(int redirects=0;redirects<=8;redirects++)
            {
                cancel.ThrowIfCancellationRequested();
                var req=(HttpWebRequest)WebRequest.Create(url);
                req.AllowAutoRedirect=false;
                req.UserAgent="ForgeNEO-LoRA/1.2";
                req.Timeout=30000; req.ReadWriteTimeout=60000;
                if(sendKey && Api(url) && key.Length>0) req.Headers[HttpRequestHeader.Authorization]="Bearer "+key;
                using(cancel.Register(req.Abort))
                {
                    HttpWebResponse response;
                    try {response=(HttpWebResponse)req.GetResponse();}
                    catch(WebException e)
                    {
                        cancel.ThrowIfCancellationRequested();
                        using(var error=e.Response as HttpWebResponse)
                        {
                            if(error==null) throw new IOException("通信エラー "+e.Status);
                            int status=(int)error.StatusCode;
                            throw new IOException("HTTP "+status+(status==401 || status==403 ? "  APIキーとモデルのダウンロード権限を確認してください。" : ""));
                        }
                    }
                    using(response)
                    {
                        int code=(int)response.StatusCode;
                        if(code>=300 && code<400)
                        {
                            Uri dest;
                            if(!Uri.TryCreate(url,response.Headers["Location"],out dest) || dest.Scheme!="https" || dest.UserInfo.Length!=0)
                                throw new InvalidDataException("不正な転送先です。");
                            sendKey=sendKey && Api(dest);
                            url=dest; continue;
                        }
                        using(var stream=response.GetResponseStream()) read(stream);
                        return;
                    }
                }
            }
            throw new IOException("転送回数が上限に達しました。");
        }
        internal static List<LoraFile> ListFiles(string key,CancellationToken cancel,Action<string> status)
        {
            string page="https://civitai.com/api/v1/models?username="+Author+"&types=LORA&limit=100&nsfw=true";
            var seen=new HashSet<string>();
            var files=new List<LoraFile>();
            while(!string.IsNullOrEmpty(page))
            {
                var uri=new Uri(page);
                if(!Api(uri) || uri.AbsolutePath!="/api/v1/models" || !seen.Add(uri.AbsoluteUri))
                    throw new InvalidDataException("一覧の次ページを確認できません。");
                status("投稿一覧を取得中　"+seen.Count+" ページ目");
                string body=null;
                ReadUrl(uri,key,cancel,s=>{using(var reader=new StreamReader(s)) body=reader.ReadToEnd();});
                files.AddRange(ParsePage(body,out page));
            }
            return files;
        }
        static string Hex(byte[] bytes) {return BitConverter.ToString(bytes).Replace("-","").ToLowerInvariant();}
        internal static bool Matches(string path,string sha,CancellationToken cancel)
        {
            using(var stream=File.OpenRead(path)) using(var hash=SHA256.Create())
            {
                var block=new byte[1024*1024]; int n;
                while((n=stream.Read(block,0,block.Length))>0)
                {cancel.ThrowIfCancellationRequested();hash.TransformBlock(block,0,n,block,0);}
                hash.TransformFinalBlock(block,0,0);return Hex(hash.Hash)==sha;
            }
        }
        internal static void SaveVerified(Stream input,string target,string sha,CancellationToken cancel)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(target));
            string temp=target+"."+Guid.NewGuid().ToString("N")+".part";
            try
            {
                using(var output=new FileStream(temp,FileMode.CreateNew,FileAccess.Write,FileShare.None))
                using(var hash=SHA256.Create())
                {
                    var block=new byte[1024*1024];int n;
                    while((n=input.Read(block,0,block.Length))>0)
                    {
                        cancel.ThrowIfCancellationRequested();
                        output.Write(block,0,n);hash.TransformBlock(block,0,n,block,0);
                    }
                    hash.TransformFinalBlock(block,0,0);
                    if(Hex(hash.Hash)!=sha) throw new InvalidDataException("SHA-256が配布情報と一致しません。");
                }
                cancel.ThrowIfCancellationRequested();
                File.Move(temp,target);
            }
            finally {if(File.Exists(temp)) File.Delete(temp);}
        }
    }
}
