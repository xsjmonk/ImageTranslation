import System.IO.File as File;
import System.IO.Directory as Dir;
import System.IO.Path as Path;
import System.String as Str;
import Ex.Console as Console;
import Ex.Powershell as Powershell;


@blue = "3399FF";
@green = "5FD7AF";
@yellow = "FF5630";
@red = "FF5C5C";

var en = TestTranslation();
mark(@blue, en);

=> null;

func TestTranslation() {
	string sentence = "加厚防水面料，耐磨耐用，适合日常使用。";
	sentence = text();
	_ clr.Ex.StatusConsole.Start("Testing clr.Ex.Http.Send...");
	var httpResult = Translate(sentence);
	_ clr.Ex.StatusConsole.Stop();

	/*_ clr.Ex.StatusConsole.Start("Testing curl.exe...");
	var curlResult = TranslateWithCurl(sentence);
	_ clr.Ex.StatusConsole.Stop();
	_ mark("70D070", "curl.exe: " & curlResult);*/

	return httpResult;
}


func Translate(sentence) {
	string api = "http://127.0.0.1:8091/translate";

	string body = "{\"text\":" & clr.Ex.Json.Serialize(sentence) & ",\"format\":\"html\"}";

	var headers = [
		new { Name = "Content-Type", Value = "application/json" }
	];

	var response = clr.Ex.Http.Send("POST", api, body, headers);

	if(!response..IsSuccessStatusCode) {
		return "HTTP " & response..StatusCode & ": " & response..Content;
	}

	var translated = clr.Ex.Json.Deserialize(
		response..Content,
		new { translation: "" }
	);

	return translated.translation;
}

func TranslateWithCurl(sentence) {
	string api = "http://127.0.0.1:8091/translate";
	string body = "{\"text\":" & clr.Ex.Json.Serialize(sentence) & "}";
	string tempFile = clr.System.IO.Path.Combine(
		clr.System.IO.Path.GetTempPath(),
		"xs-translate-" & clr.System.Guid.NewGuid().ToString("N") & ".json"
	);

	_ clr.System.IO.File.WriteAllText(
		tempFile,
		body,
		new clr.System.Text.UTF8Encoding(false)
	);

	StringBuilder command =<<<
curl.exe --silent --show-error --fail-with-body --request POST --header "Content-Type: application/json" --data-binary "@__BODY_FILE__" "__API__"
>>>;

	_ command.Replace("__BODY_FILE__", tempFile);
	_ command.Replace("__API__", api);

	string responseText = "";

	try {
		responseText = RunPowershellFromMemory(command.ToString(), true);
	}
	catch {
		responseText = "curl failed";
	}

	if(clr.System.IO.File.Exists(tempFile)) {
		clr.System.IO.File.Delete(tempFile);
	}

	if(responseText == "curl failed") {
		return responseText;
	}

	var translated = clr.Ex.Json.Deserialize(
		responseText,
		new { translation: "" }
	);

	return translated.translation;
}



func RunPowershellFromMemory(command, showError) {
	var p = new clr.System.Diagnostics.Process();
	p.StartInfo.WindowStyle = clr.System.Diagnostics.ProcessWindowStyle.Minimized;
	p.StartInfo.CreateNoWindow = true;
	p.StartInfo.UseShellExecute = false;
	p.StartInfo.RedirectStandardOutput = true;
	p.StartInfo.RedirectStandardError = true;
	p.StartInfo.FileName = "powershell.exe";
	p.StartInfo.ArgumentList.Add("-NoLogo");
	p.StartInfo.ArgumentList.Add("-NoProfile");
	p.StartInfo.ArgumentList.Add("-NonInteractive");
	p.StartInfo.ArgumentList.Add("-ExecutionPolicy");
	p.StartInfo.ArgumentList.Add("Bypass");
	p.StartInfo.ArgumentList.Add("-Command");
	p.StartInfo.ArgumentList.Add("& {" & command & "}");
	p.Start();

	string stderrx = p.StandardError.ReadToEnd();
	var outputStream = p.StandardOutput.BaseStream;
	var ms = new clr.System.IO.MemoryStream();
	outputStream.CopyTo(ms);

	p.WaitForExit();
	if(!clr.System.String.IsNullOrEmpty(stderrx) && (bool)showError) { 
		mark("F65B3B", stderrx);
	}
	p.Dispose();
	outputStream.Dispose();
	string output = clr.System.Text.Encoding.UTF8.GetString(ms.ToArray());
	ms.Dispose();

	return output;
}


void mark(color, content) {
	clr.Ex.Console.Markup("[#" & color & "]"
		& content.ToString().Replace("[", "").Replace("]", "").Replace("[/]", "")
		& "[/]" & clr.System.Environment.NewLine
	);
}


func text() {
=> <<<
这是一款面向日常办公、旅行和家庭使用场景设计的多功能充电设备。产品支持 USB-C Power Delivery，并兼容多种主流设备，包括 iPhone 16 Pro、iPad Pro、MacBook Air、Samsung Galaxy S25 以及部分支持 USB PD 的 Windows laptop。单口输出时，USB-C1 最高可提供 100W 功率，USB-C2 最高可提供 65W，USB-A 接口则适合连接传统设备，例如耳机、智能手表和较老型号的手机。

在实际使用过程中，设备会根据连接终端的需求自动调整输出功率。例如，当 MacBook Air 连接 USB-C1，而 iPhone 16 Pro 同时连接 USB-C2 时，系统会动态分配功率，以尽量保证两台设备都能保持稳定充电。对于不支持快速充电协议的设备，充电器会自动回落到标准电压和电流，避免因为错误的功率输出而影响设备正常工作。

产品内部采用 GaN technology，也就是氮化镓功率器件。与传统硅基充电器相比，GaN 方案通常可以在更小的体积下实现较高的功率密度，同时降低部分高负载情况下的能量损耗。需要注意的是，设备在持续高功率工作时仍然会产生热量，因此使用过程中应保证周围具有正常空气流通条件，不建议长期覆盖在被褥、衣物或其他不利于散热的材料下面。

为了提高使用安全性，设备集成了多项保护机制，包括 Over-Voltage Protection、Over-Current Protection、Short-Circuit Protection 和 Temperature Protection。当内部温度超过预设范围时，控制系统可能主动降低输出功率。此时用户可能会发现充电速度暂时下降，这属于正常保护行为，并不一定意味着产品发生故障。待温度下降后，系统会根据实际情况恢复正常输出。

产品外壳使用阻燃材料，并在结构设计中考虑了日常跌落、插拔和运输过程中的机械应力。不过，“阻燃”并不等于“不会燃烧”，也不代表设备可以在极端环境中使用。请勿将产品放置在明火附近，也不要在明显进水、外壳严重破损或接口已经变形的情况下继续使用。如果发现异常气味、明显异响、过度发热或反复断电，应立即停止使用并检查连接设备和电源环境。

>>>;

}
