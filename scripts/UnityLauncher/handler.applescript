-- Registered handler for unity3dgen:// URIs.
-- Dispatches to Contents/Resources/handler.sh inside the .app bundle.
on open location this_URL
    set myPath to POSIX path of (path to me)
    set handlerPath to myPath & "Contents/Resources/handler.sh"
    -- Expand PATH so Homebrew python3/tools are visible inside do shell script,
    -- which runs with a minimal environment (/usr/bin:/bin only).
    set expandedPath to "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    try
        do shell script "export PATH=" & quoted form of expandedPath & ":\"$PATH\"; " & quoted form of handlerPath & " " & quoted form of this_URL
    on error errMsg number errNum
        display dialog "Unity launcher error (" & errNum & "): " & errMsg buttons {"OK"} default button 1
    end try
end open location
