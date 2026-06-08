-- Registered handler for unity3dgen:// URIs.
-- Dispatches to Contents/Resources/handler.sh inside the .app bundle.
on open location this_URL
    set myPath to POSIX path of (path to me)
    set handlerPath to myPath & "Contents/Resources/handler.sh"
    try
        do shell script quoted form of handlerPath & " " & quoted form of this_URL
    on error errMsg number errNum
        display dialog "Unity launcher error: " & errMsg buttons {"OK"} default button 1
    end try
end open location
