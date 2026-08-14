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

该产品支持输入电压 AC 100-240V, 50/60Hz，因此通常可以在多数国家和地区使用。但是，不同国家的插头标准不同，出国旅行时可能仍然需要额外的 travel adapter。转换插头只改变物理接口形状，并不会自动完成电压转换，因此在连接不支持宽电压输入的其他电器时，仍然需要确认设备铭牌参数。

在数据线选择方面，如果希望实现 100W USB-C 输出，应使用支持相应功率等级的 USB-C to USB-C cable。部分较老或规格较低的数据线可能只能支持 60W，甚至更低的功率。对于支持 E-Marker 的高功率线材，设备能够更准确地识别线材能力并协商合适的充电参数。若数据线本身质量较差，即使充电器和终端设备都支持快速充电，也可能无法达到预期速度。

例如，一台标称支持 65W charging 的笔记本电脑，并不意味着任何 USB-C 线缆都能够稳定提供 65W。实际结果取决于 charger、cable 和 device 三者之间的协议协商。如果其中任一环节不兼容，系统通常会采用较低的安全功率。因此，在排查充电速度问题时，应分别检查电源、数据线以及终端设备，而不是只判断充电器本身。

对于手机用户来说，充电功率也会受到电池温度、电量百分比和系统策略影响。很多智能手机在低电量时会使用较高功率快速充电，而在电池接近充满时逐渐降低功率。这种现象属于 Battery Management System 的正常工作逻辑。即使充电器标称 65W 或 100W，也不意味着手机在整个充电过程中都会持续使用最大功率。

设备表面的 LED indicator 用于显示当前工作状态。正常连接电源后，指示灯会亮起；某些异常保护状态下，指示灯可能闪烁或暂时关闭。不同批次产品的指示方式可能略有差异，因此最终应以正式说明书为准。不要仅凭指示灯状态判断内部电子系统是否完全正常。

在日常携带时，建议避免将充电器和钥匙、硬币或其他金属物品混放，以减少接口被刮伤或异物进入的可能性。如果 USB-C 接口内部存在灰尘或纤维，可以在断电状态下使用适当方式进行清理，但不建议使用金属针状物直接插入接口内部。错误操作可能损坏触点，并增加短路风险。

对于长期不用的设备，建议存放在干燥、通风、避免阳光长期直射的位置。不要将其放置在高湿度环境中，也不要长期置于汽车仪表台等高温区域。电子产品的寿命不仅与内部元件质量有关，也会受到温度、湿度、使用频率和负载条件影响。

从实际体验来看，这类多口高功率充电器最大的优势不是单纯追求最高 wattage，而是减少出行时需要携带的电源数量。一只体积相对紧凑的 charger 可以同时服务 laptop、tablet、phone 和 wearable device，对于经常出差或者需要在办公室使用多台电子设备的人来说更加方便。

但是，在多人同时使用或者多个高功率设备同时连接时，应注意总输出功率限制。例如设备标称总功率为 120W，并不代表每个接口都可以同时输出 100W。不同端口组合通常具有不同的 power allocation strategy。具体功率分配规则应参考产品规格表，例如 USB-C1 + USB-C2 可能采用 65W + 45W，而 USB-C1 + USB-C2 + USB-A 同时使用时，则可能进一步调整。



>>>;

}
