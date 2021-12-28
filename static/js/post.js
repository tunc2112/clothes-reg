// merge toc.js and disqus.js into post.js

function create_toc() {
	var toc_div = document.getElementById("toc");
	var post_content = document.getElementById("post_left_side");
	var all_headings = post_content.querySelectorAll("h1,h2,h3,h4,h5,h6");
	for (var heading of all_headings) {
		var link_str = heading.textContent.toLowerCase().split(" ").join("-");
		// console.log(heading, link_str);
		heading.setAttribute("id", link_str);

		var anchor_element = document.createElement("A");
		anchor_element.innerHTML = heading.textContent;
		anchor_element.href = "#" + link_str;
		anchor_element.className = "toc_anchor toc_anchor_" + heading.tagName.toLowerCase();

		var parent_anchor = document.createElement("DIV");
		parent_anchor.appendChild(anchor_element);

		// toc_div.appendChild(anchor_element);
		toc_div.appendChild(parent_anchor);
	}
}

function embed_disqus() {
	/**
	*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
	*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables
	*/
	var disqus_config = function () {
		this.page.url = "https://tunc2112.me/{{ url_for('posts') }}/{{ post.path }}";  // Replace PAGE_URL with your page's canonical URL variable
		this.page.identifier = "{{ post.path }}"; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
	};

	(function() { // DON'T EDIT BELOW THIS LINE
	var d = document, s = d.createElement('script');
	s.src = 'https://tunc2112.disqus.com/embed.js';
	s.setAttribute('data-timestamp', +new Date());
	(d.head || d.body).appendChild(s);
	})();
}
