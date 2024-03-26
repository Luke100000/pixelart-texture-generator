from streamlit.components.v1 import html


def copy_mime_button(label: str, data: bytes, mime_type: str):
    data = f"new Uint8Array({list(data)});"
    html(
        """
        <button id="copyButton">"""
        + label
        + """</button>
        <script>
        function toClipboard() {
            const image = """
        + data
        + """;
            const blob = new Blob([ image ], { type : \""""
        + mime_type
        + """\" });
            const data = [new ClipboardItem({ [blob.type]: blob })];
            navigator.clipboard.write(data).then(function() {
                console.log('Async: Copying to clipboard was successful!');
            }, function(err) {
                console.error('Async: Could not copy text: ', err);
            })
        }
        document.getElementById("copyButton").onclick = toClipboard;
        """, height=32
    )
