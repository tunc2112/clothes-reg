function convert_emoji() {
    let body = document.body;
    let s = body.innerHTML;
    // var s = ":joy: :smile: :grin:"; // body.innerHTML;
    // console.log(s.match(/:[\w-]+:/g));
    body.innerHTML = s.replace(/:[\w-]+?:/g, x => twemoji.convert.fromCodePoint(emoji_unicode[x]));
    twemoji.parse(body);
}
